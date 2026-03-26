import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class AsymmetricDownsample(nn.Module):
    """
    Asymmetric downsampling block.

    Motivation:
        This module was introduced to preserve more horizontal details while
        applying more aggressive compression along the vertical dimension,
        which is intuitively suitable for handwritten text line images.

    Design:
        - First apply a horizontal convolution: (1x3, stride=(1, r_h))
        - Then apply a vertical convolution:   (3x1, stride=(r_v, 1))

    Experimental observation:
        Although this asymmetric design can reduce model parameters and
        slightly improve compactness, our experiments show that it tends to
        degrade recognition performance compared with the default symmetric
        setting.

    Recommendation:
        For reproducible experiments and the main results reported in the paper,
        training should use the default non-asymmetric setting, i.e. enable
        `--no-aniso` unless this module is being explicitly evaluated in an
        ablation study.
    """
    def __init__(self, in_channels, out_channels,
                 horizontal_ratio=2, vertical_ratio=4):
        super().__init__()
        self.horizontal_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 3),
            stride=(1, horizontal_ratio),
            padding=(0, 1),
            bias=False,
        )
        self.vertical_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(3, 1),
            stride=(vertical_ratio, 1),
            padding=(1, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.horizontal_conv(x)
        x = self.vertical_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CSPDown(nn.Module):
    def __init__(self, in_channels, out_channels, use_asymmetric=False):
        super().__init__()
        self.part1 = nn.Conv2d(in_channels, out_channels // 2,
                               kernel_size=1, bias=False)
        self.part2_conv = DoubleConv(in_channels, out_channels // 2)
        if use_asymmetric:
            self.down = AsymmetricDownsample(out_channels // 2,
                                             out_channels // 2)
        else:
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                DoubleConv(out_channels // 2, out_channels // 2),
            )
        self.merge_conv = nn.Conv2d(out_channels, out_channels,
                                    kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        part1 = self.part1(x)
        part2 = self.part2_conv(x)
        part2 = self.down(part2)
        if part1.shape[2:] != part2.shape[2:]:
            part1 = F.adaptive_avg_pool2d(part1, part2.shape[2:])

        x = torch.cat([part1, part2], dim=1)
        x = self.merge_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PlainDown(nn.Module):
    """
    Plain downsampling block used only for ablation.

    Purpose:
        This module is provided as a simplified replacement for the CSP-based
        downsampling path in `CSPUNetBackbone`, in order to evaluate the
        contribution of CSP-style feature partitioning and fusion.

    Variants:
        - use_asymmetric=True:
            Use `AsymmetricDownsample`
        - use_asymmetric=False:
            Use standard MaxPool + DoubleConv

    Note:
        This block is not part of the default HTR-HSS configuration and is
        retained only for controlled ablation experiments. It can be safely
        ignored in normal training and usage.
    """
    def __init__(self, in_channels, out_channels, use_asymmetric=False):
        super().__init__()
        if use_asymmetric:
            self.down = AsymmetricDownsample(in_channels, out_channels)
        else:
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                DoubleConv(in_channels, out_channels),
            )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, x1_channels, x2_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                x1_channels, x1_channels // 2,
                kernel_size=2, stride=2
            )
        self.conv = DoubleConv(x1_channels + x2_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CSPUNetBackbone(nn.Module):
    def __init__(self,
                 n_channels: int = 1,
                 n_classes: int = 1,
                 bilinear: bool = True,
                 use_asymmetric: bool = False,
                 use_csp: bool = True,
                 base_channels: int = 32,
                 num_levels: int = 4,
                 channel_multiplier: float = 1.5):
        super().__init__()

        self.inc = DoubleConv(n_channels, base_channels)
        self.downs = nn.ModuleList()

        DownBlock = CSPDown if use_csp else PlainDown

        for i in range(num_levels - 1):
            in_ch = int(base_channels * (channel_multiplier ** i))
            out_ch = int(base_channels * (channel_multiplier ** (i + 1)))
            self.downs.append(
                DownBlock(in_ch, out_ch, use_asymmetric=use_asymmetric)
            )

        self.ups = nn.ModuleList()
        for i in range(num_levels - 1, 0, -1):
            in_ch = int(base_channels * (channel_multiplier ** i))
            skip_ch = int(base_channels * (channel_multiplier ** (i - 1)))
            out_ch = skip_ch
            self.ups.append(Up(in_ch, skip_ch, out_ch, bilinear))

        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        encoder_feats = [self.inc(x)]
        for down in self.downs:
            encoder_feats.append(down(encoder_feats[-1]))

        x = encoder_feats[-1]
        for i, up in enumerate(self.ups):
            x = up(x, encoder_feats[-(i + 2)])

        x = self.out_conv(x)
        return x
