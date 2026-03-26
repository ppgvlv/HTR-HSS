# ==========================================================
# HTRBiMambaHybrid
# ==========================================================
#
# Main architecture of HTR-HSS for offline handwritten text recognition.
#
# Overall pipeline:
#   CSP-UNetBackbone
#     -> TemporalDownsample
#     -> [Bi-Mamba + (sparse) Self-Attention + MLP] x depth
#     -> LayerNorm
#     -> Linear classifier
#
# Main interfaces:
#   - forward(x, mask_ratio, max_span_length, use_masking)
#       -> logits of shape [B, T, V]
#
#   - create_model(nb_cls, img_size, **kwargs)
#       -> factory function for model construction
# Base Model:
#   A0:  noAniso   -> use_asymmetric=False
#
# Ablation switches used in the paper:
#   A1:  noCSP     -> use_csp=False
#   A2:  noAttn    -> enable_attn=False
#   A3:  noMamba   -> enable_mamba=False
#
# Hyperparameter studies:
#   A4-A7:   td_stride   in {2, 4, 6, 8}
#   A8-A12:  attn_heads  in {1, 2, 4, 8, 16}
#   A13-A17: attn_window in {None, 16, 32, 64, 128}
#   A18-A21: attn_every  in {1, 2, 3, 4}
#   A22-A26: depth       in {1, 2, 4, 6, 8}
#
# Notes:
#   - This file contains both the default model configuration and several
#     optional variants retained for controlled ablation experiments.
#   - The default paper configuration corresponds to the main HTR-HSS model.
#
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mamba_ssm import Mamba

from model.backbone_csp_unet import CSPUNetBackbone

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x / keep_prob * random_tensor
        return output


class TemporalDownsample(nn.Module):
    """
      in： [B, C, T]
      out： [B, T', C_out]
    """
    def __init__(self, in_dim, out_dim, stride=4):
        super().__init__()
        self.proj = nn.Conv1d(
            in_dim, out_dim,
            kernel_size=stride,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, C, T]
        x = self.proj(x)          # [B, C, T']
        x = x.permute(0, 2, 1)    # [B, T', C]
        x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BiMambaLayer(nn.Module):
    def __init__(self, dim, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.mamba_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self.mamba_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        else:
            self.mamba_fwd = None
            self.mamba_bwd = None

    def forward(self, x):  # [B, T, C]
        if not self.enabled:
            return x
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(x.flip(1)).flip(1)
        return fwd + bwd


class TinySelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4,
                 attn_drop=0., proj_drop=0.,
                 window_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # [B, T, C]
        if self.window_size is None or self.window_size >= x.size(1):
            out, _ = self.attn(x, x, x, need_weights=False)
            return self.proj_drop(out)
        B, T, C = x.shape
        W = self.window_size
        chunks = []
        for s in range(0, T, W):
            e = min(s + W, T)
            xw = x[:, s:e, :]
            ow, _ = self.attn(xw, xw, xw, need_weights=False)
            chunks.append(ow)
        out = torch.cat(chunks, dim=1)
        return self.proj_drop(out)


class HybridBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop_path=0.1,
                 attn_heads=4,
                 attn_window=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 enable_mamba: bool = True,
                 enable_attn: bool = True):
        super().__init__()

        self.enable_mamba = enable_mamba
        self.enable_attn = enable_attn

        self.norm1 = norm_layer(dim, elementwise_affine=True)
        self.mamba = BiMambaLayer(dim, enabled=enable_mamba)
        self.dp1 = DropPath(drop_path) if enable_mamba else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        if enable_attn:
            self.attn = TinySelfAttention(
                dim,
                num_heads=attn_heads,
                attn_drop=0.,
                proj_drop=0.,
                window_size=attn_window,
            )
            self.dp2 = DropPath(drop_path)
        else:
            self.attn = None
            self.dp2 = nn.Identity()

        self.norm3 = norm_layer(dim, elementwise_affine=True)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio,
                       act_layer=act_layer, drop=0.)
        self.dp3 = DropPath(drop_path)

    def forward(self, x):  # [B, T, C]
        if self.enable_mamba:
            x = x + self.dp1(self.mamba(self.norm1(x)))
        if self.enable_attn and self.attn is not None:
            x = x + self.dp2(self.attn(self.norm2(x)))
        x = x + self.dp3(self.mlp(self.norm3(x)))
        return x


class HTRBiMambaHybrid(nn.Module):
    def __init__(self,
                 nb_cls=90,
                 img_size=(512, 64),
                 embed_dim=128,
                 depth=6,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_path_rate=0.1,
                 num_levels=4,
                 channel_multiplier=1.5,
                 td_stride=4,
                 attn_every=3,
                 attn_heads=4,
                 attn_window=None,
                 # Ablation：
                 use_asymmetric: bool = True,   # A0: noAniso -> False
                 use_csp: bool = True,          # A1: noCSP   -> False
                 enable_mamba: bool = True,     # A3: noMamba -> False
                 enable_attn: bool = True):     # A2: noAttn  -> False
        super().__init__()

        self.patch_embed = CSPUNetBackbone(
            n_channels=1,
            n_classes=embed_dim,
            bilinear=True,
            use_asymmetric=use_asymmetric,
            use_csp=use_csp,
            base_channels=embed_dim,
            num_levels=num_levels,
            channel_multiplier=channel_multiplier,
        )

        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.temporal_down = TemporalDownsample(
            embed_dim, embed_dim, stride=td_stride
        )

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        blocks = []
        for i in range(depth):
            use_attn_layer = (
                attn_every is not None and attn_every > 0
                and (i + 1) % attn_every == 0
            )
            blocks.append(
                HybridBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    attn_heads=attn_heads if use_attn_layer else max(1, attn_heads // 2),
                    attn_window=attn_window if use_attn_layer else attn_window,
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    enable_mamba=enable_mamba,
                    enable_attn=enable_attn,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.norm = (norm_layer(embed_dim, eps=1e-6)
                     if norm_layer is nn.LayerNorm
                     else norm_layer(embed_dim))
        self.head = nn.Linear(embed_dim, nb_cls)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    # -------- span masking --------
    def _generate_span_mask(self, x, mask_ratio, max_span_length):
        B, T, C = x.shape
        mask = torch.ones(B, T, 1, device=x.device)
        total = int(T * mask_ratio)
        num_spans = max(1, total // max_span_length)
        for _ in range(num_spans):
            if T - max_span_length <= 0:
                break
            s = torch.randint(0, T - max_span_length + 1, (1,), device=x.device)
            mask[:, s:s + max_span_length, :] = 0
        return mask

    def _random_masking(self, x, mask_ratio, max_span_length):
        mask = self._generate_span_mask(x, mask_ratio, max_span_length)
        return x * mask + (1 - mask) * self.mask_token

    def forward(self, x, mask_ratio=0.0,
                max_span_length=1, use_masking=False):

        feats = self.patch_embed(x)                                # [B, C, H', W']
        feats = F.adaptive_avg_pool2d(feats, (1, feats.size(-1)))  # [B, C, 1, W]
        feats = feats.squeeze(2)                                   # [B, C, W]

        x = self.temporal_down(feats)                              # [B, T', C]

        if use_masking and mask_ratio > 0:
            x = self._random_masking(x, mask_ratio, max_span_length)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        logits = self.head(x)                                      # [B, T', V]
        return logits

    
def create_model(nb_cls, img_size, **kwargs):
    raw_attn_window = kwargs.pop("attn_window", None)

    # Unified mapping rule:
    #   None       -> None     (not provided by caller; use default, equivalent to global attention)
    #   <= 0       -> None     (-1 / 0 are treated as "global attention")
    #   > 0        -> integer  (use local attention with specified window size)
    if raw_attn_window is None or raw_attn_window <= 0:
        attn_window = None
    else:
        attn_window = raw_attn_window

    model = HTRBiMambaHybrid(
        nb_cls=nb_cls,
        img_size=img_size,
        embed_dim=kwargs.pop("embed_dim", 128),
        depth=kwargs.pop("depth", 6),
        mlp_ratio=kwargs.pop("mlp_ratio", 4.0),
        norm_layer=kwargs.pop("norm_layer", partial(nn.LayerNorm, eps=1e-6)),
        drop_path_rate=kwargs.pop("drop_path_rate", 0.1),
        num_levels=kwargs.pop("num_levels", 4),
        channel_multiplier=kwargs.pop("channel_multiplier", 1.5),
        td_stride=kwargs.pop("td_stride", 8),
        attn_every=kwargs.pop("attn_every", 3),
        attn_heads=kwargs.pop("attn_heads", 4),
        attn_window=attn_window,
        use_asymmetric=kwargs.pop("use_asymmetric", False),
        use_csp=kwargs.pop("use_csp", True),
        enable_mamba=kwargs.pop("enable_mamba", True),
        enable_attn=kwargs.pop("enable_attn", True),
    )
    return model

