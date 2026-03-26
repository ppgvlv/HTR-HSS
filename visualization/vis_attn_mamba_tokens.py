#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme 2 (v4, paper-friendly): 3x2 grid per dataset.

For EACH dataset, we take 2 specified test images and generate ONE figure:
  rows:
    1) Original Image
    2) Self-Attention (last block)
    3) BiMamba dependency (last block)
  cols: the 2 chosen images.

This version is modified for paper-friendly export:
- Supports vector outputs: PDF / SVG
- Also optionally supports PNG
- Default export formats: pdf,svg

Run examples (from repo root):
  python3 vis/vis_attn_mamba_tokens_v4.py --vis-dir vis/attn_mamba_final_v4 --exp-name IAM_A0_noAniso --td-stride 4 --no-aniso IAM
  python3 vis/vis_attn_mamba_tokens_v4.py --vis-dir vis/attn_mamba_final_v4 --exp-name LAM_A0_noAniso --td-stride 8 --no-aniso LAM
  python3 vis/vis_attn_mamba_tokens_v4.py --vis-dir vis/attn_mamba_final_v4 --exp-name READ_A0_noAniso --td-stride 8 --no-aniso READ

Optional:
  --save-formats pdf,svg
  --save-formats pdf
  --save-formats svg,png
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")

# Paper-friendly font embedding for PDF/PS
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt


# -----------------------------
# Repo root & imports
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import option
from model.htr_bimamba_hybrid import create_model


# -----------------------------
# Dataset paths
# -----------------------------
DATASET_DIR_MAP = {
    "IAM": "data/iam",
    "LAM": "data/LAM",
    "READ": "data/read2016",
}

DEFAULT_IMG_SELECTION = {
    "IAM": ["test_0.png"],
    "LAM": ["002_02_00.jpg"],
    "READ": ["Seite0001_1.png"],
}


def _dataset_paths(dataset: str) -> Tuple[Path, Path]:
    if dataset not in DATASET_DIR_MAP:
        raise ValueError(f"Unsupported dataset '{dataset}', expected one of {list(DATASET_DIR_MAP.keys())}")
    ds_dir = REPO_ROOT / DATASET_DIR_MAP[dataset]
    lines_dir = ds_dir / "lines"
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {ds_dir}")
    if not lines_dir.exists():
        raise FileNotFoundError(f"Lines dir not found: {lines_dir}")
    return ds_dir, lines_dir


# -----------------------------
# Model loading (mirror test.py)
# -----------------------------
def _build_model_from_opt_args(opt_args: argparse.Namespace) -> torch.nn.Module:
    model = create_model(
        nb_cls=opt_args.nb_cls,
        img_size=tuple(opt_args.img_size[::-1]),  # [W,H] -> (H,W)
        embed_dim=opt_args.embed_dim,
        depth=opt_args.encoder_depth,
        mlp_ratio=opt_args.mlp_ratio,
        drop_path_rate=opt_args.drop_path_rate,
        num_levels=opt_args.num_levels,
        channel_multiplier=opt_args.channel_multiplier,
        td_stride=opt_args.td_stride,
        attn_every=opt_args.attn_every,
        attn_heads=opt_args.attn_heads,
        attn_window=opt_args.attn_window,
        use_asymmetric=not opt_args.no_aniso,
        use_csp=not opt_args.no_csp,
        enable_mamba=not opt_args.no_mamba,
        enable_attn=not opt_args.no_attn,
    )
    return model


def _load_best_ckpt(model: torch.nn.Module, out_dir: str, exp_name: str, device: torch.device, strict: bool = False) -> Path:
    ckpt_path = Path(out_dir) / exp_name / "best_CER.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)  # type: ignore
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    state = None
    if isinstance(ckpt, dict):
        if "state_dict_ema" in ckpt and isinstance(ckpt["state_dict_ema"], dict):
            state = ckpt["state_dict_ema"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]

    if state is None:
        raise RuntimeError(
            f"Unrecognized checkpoint format: {type(ckpt)} "
            f"keys={list(ckpt.keys()) if isinstance(ckpt, dict) else None}"
        )

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if strict and (len(missing) or len(unexpected)):
        raise RuntimeError(f"strict load failed: missing={missing}, unexpected={unexpected}")

    model.to(device)
    model.eval()
    return ckpt_path


# -----------------------------
# Preprocess
# -----------------------------
def _load_images(img_path: Path, target_wh: Tuple[int, int]) -> Tuple[Image.Image, torch.Tensor]:
    """
    Returns:
      pil_orig: original resolution (for display)
      x_img: [1,1,H,W] float32 resized to target_wh for model
    """
    pil_orig = Image.open(img_path).convert("L")
    W, H = target_wh
    pil_model = pil_orig.resize((W, H), resample=Image.BILINEAR)
    arr = np.array(pil_model, dtype=np.float32) / 255.0
    x_img = torch.from_numpy(arr)[None, None, :, :]
    return pil_orig, x_img


# -----------------------------
# Tokens & blocks
# -----------------------------
@torch.no_grad()
def _extract_tokens_before_blocks(model: torch.nn.Module, x_img: torch.Tensor) -> torch.Tensor:
    feats = model.patch_embed(x_img)  # [B,C,H',W']
    feats = F.adaptive_avg_pool2d(feats, (1, feats.size(-1)))  # [B,C,1,W']
    feats = feats.squeeze(2)  # [B,C,W']
    x_seq = model.temporal_down(feats)  # [B,T',C]
    return x_seq


@torch.no_grad()
def _forward_to_block_input(model: torch.nn.Module, x_seq: torch.Tensor, block_idx: int) -> torch.Tensor:
    x = x_seq
    for i in range(block_idx):
        x = model.blocks[i](x)
    return x


# -----------------------------
# Attention map
# -----------------------------
@torch.no_grad()
def _compute_attention_map_from_block(block: torch.nn.Module, x_block_in: torch.Tensor) -> Optional[np.ndarray]:
    if not getattr(block, "enable_attn", False):
        return None
    if getattr(block, "attn", None) is None:
        return None

    tiny = block.attn
    mha = tiny.attn
    window_size = getattr(tiny, "window_size", None)

    x2 = block.norm2(x_block_in)  # [B,T,C]
    B, T, _ = x2.shape

    def _mha_weights(xq: torch.Tensor) -> torch.Tensor:
        _, w = mha(xq, xq, xq, need_weights=True, average_attn_weights=False)
        return w  # [B, heads, T, T]

    if window_size is None or int(window_size) >= T:
        w = _mha_weights(x2)
        w_avg = w.mean(dim=1)[0]
        w_np = w_avg.detach().cpu().float().numpy()
        w_np = w_np / (w_np.max() + 1e-12)
        return w_np

    W = int(window_size)
    full = torch.zeros((T, T), device=x2.device, dtype=torch.float32)
    for s in range(0, T, W):
        e = min(s + W, T)
        xw = x2[:, s:e, :]
        ww = _mha_weights(xw).mean(dim=1)[0]  # [w,w]
        full[s:e, s:e] = ww
    full = full / (full.max() + 1e-12)
    return full.detach().cpu().numpy()


# -----------------------------
# BiMamba dependency
# -----------------------------
@torch.no_grad()
def _compute_mamba_dependency_map_from_block(block: torch.nn.Module, x_block_in: torch.Tensor, eps: float) -> Optional[np.ndarray]:
    if not getattr(block, "enable_mamba", False):
        return None
    if getattr(block, "mamba", None) is None:
        return None

    mamba_layer = block.mamba
    x1 = block.norm1(x_block_in)  # [B,T,C]
    B, T, C = x1.shape
    if B != 1:
        raise RuntimeError("This visualization expects batch=1")

    base = mamba_layer(x1)  # [1,T,C]
    D = torch.zeros((T, T), device=x1.device, dtype=torch.float32)

    for j in range(T):
        x_pert = x1.clone()
        dir_vec = torch.sign(x1[:, j, :])
        if torch.all(dir_vec == 0):
            dir_vec[:, 0] = 1.0
        dir_vec = dir_vec / (dir_vec.norm(dim=-1, keepdim=True) + 1e-12)
        x_pert[:, j, :] = x_pert[:, j, :] + eps * dir_vec
        out = mamba_layer(x_pert)
        delta = (out - base).norm(dim=-1)[0]  # [T]
        D[:, j] = delta

    D = D / (D.max() + 1e-12)
    return D.detach().cpu().numpy()


# -----------------------------
# Save helper
# -----------------------------
def _save_figure_multi_formats(
    fig: plt.Figure,
    out_stem: Path,
    save_formats: List[str],
) -> List[Path]:
    saved_paths = []
    valid_formats = {"pdf", "svg", "png"}

    for fmt in save_formats:
        fmt = fmt.lower().strip()
        if fmt not in valid_formats:
            raise ValueError(f"Unsupported save format: {fmt}. Choose from {sorted(valid_formats)}")

        out_path = out_stem.with_suffix(f".{fmt}")

        if fmt in {"pdf", "svg"}:
            fig.savefig(
                out_path,
                format=fmt,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        elif fmt == "png":
            fig.savefig(
                out_path,
                format="png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
            )

        saved_paths.append(out_path)

    return saved_paths

# -----------------------------
# Plotting: 3x1 grid
# -----------------------------

def _plot_grid_3x1(
    out_stem: Path,
    dataset_tag: str,
    img_name: str,
    pil_img: Image.Image,
    attn_map: Optional[np.ndarray],
    dep_map: Optional[np.ndarray],
    save_formats: List[str],
):
    """
    3 rows × 1 column:
      1) Original
      2) Attention
      3) Mamba
    """

    fig = plt.figure(figsize=(6.5, 9), dpi=300)

    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.5, 1.2, 1.2],
        hspace=0.35,
        left=0.12,
        right=0.95,
        top=0.95,
        bottom=0.06,
    )

    # ---------- Row 1 ----------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(
        pil_img,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )
    ax1.set_title("Original Image", fontsize=13, pad=6)
    ax1.set_axis_off()

    # ---------- Row 2 ----------
    ax2 = fig.add_subplot(gs[1, 0])

    if attn_map is not None:
        im2 = ax2.imshow(
            attn_map,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            interpolation="nearest",
        )
        ax2.set_title("Self-Attention (last block)", fontsize=13)
        ax2.set_xlabel("Key token index (T')")
        ax2.set_ylabel("Query token index (T')")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    else:
        ax2.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax2.set_axis_off()

    # ---------- Row 3 ----------
    ax3 = fig.add_subplot(gs[2, 0])

    if dep_map is not None:
        im3 = ax3.imshow(
            dep_map,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax3.set_title("BiMamba dependency (last block)", fontsize=13)
        ax3.set_xlabel("Perturbed token index (T')")
        ax3.set_ylabel("Output token index (T')")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax3.set_axis_off()

    os.makedirs(out_stem.parent, exist_ok=True)
    saved_paths = _save_figure_multi_formats(fig, out_stem, save_formats)
    plt.close(fig)
    return saved_paths

# -----------------------------
# Wrapper args
# -----------------------------
def _parse_wrapper_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    w = argparse.ArgumentParser(add_help=False)
    w.add_argument(
        "--vis-dir",
        type=str,
        default="vis/attn_mamba_final_v4",
        help="Output folder (relative to repo root ok).",
    )
    w.add_argument(
        "--block-idx",
        type=int,
        default=-1,
        help="Which Hybrid block index to visualize (-1=last).",
    )
    w.add_argument(
        "--eps",
        type=float,
        default=1e-2,
        help="Finite-diff epsilon for BiMamba dependency.",
    )
    w.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda / cpu",
    )
    w.add_argument(
        "--strict-load",
        action="store_true",
        help="Strictly enforce checkpoint key matching.",
    )
    w.add_argument(
        "--imgs",
        type=str,
        default=None,
        help="Override 2 images, comma-separated names (within dataset lines/).",
    )
    w.add_argument(
        "--save-formats",
        type=str,
        default="pdf,svg",
        help="Comma-separated output formats, e.g. pdf,svg or pdf or pdf,svg,png",
    )
    return w.parse_known_args(argv)


def main():
    wargs, remaining = _parse_wrapper_args(sys.argv[1:])

    # Let project option parse remaining args (expects dataset subcommand at end)
    sys.argv = [sys.argv[0]] + remaining
    opt_args = option.get_args_parser()

    dataset = getattr(opt_args, "subcommand", None) or getattr(opt_args, "dataset", None) or getattr(opt_args, "data_name", None)
    if dataset is None:
        raise RuntimeError("Cannot find dataset subcommand in parsed args. Ensure you pass e.g. 'IAM' at the end.")
    dataset = str(dataset).upper()
    if dataset == "READ2016":
        dataset = "READ"

    _, lines_dir = _dataset_paths(dataset)

    # Select 2 images
    if wargs.imgs:
        img_names = [x.strip() for x in wargs.imgs.split(",") if x.strip()]
        if len(img_names) != 1:
            raise ValueError("--imgs must contain exactly 1 names, e.g. --imgs a.png,b.png")
    else:
        if dataset not in DEFAULT_IMG_SELECTION:
            raise RuntimeError(f"No default image selection for dataset={dataset}")
        img_names = DEFAULT_IMG_SELECTION[dataset]

    # Parse save formats
    save_formats = [x.strip().lower() for x in wargs.save_formats.split(",") if x.strip()]
    if len(save_formats) == 0:
        raise ValueError("--save-formats cannot be empty")

    # Device
    device = torch.device(wargs.device if (wargs.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Model
    model = _build_model_from_opt_args(opt_args)
    _ = _load_best_ckpt(model, opt_args.out_dir, opt_args.exp_name, device=device, strict=wargs.strict_load)

    # Determine block index
    block_idx = int(wargs.block_idx)
    if block_idx < 0:
        block_idx = len(model.blocks) - 1
    if block_idx < 0 or block_idx >= len(model.blocks):
        raise ValueError(f"Invalid block_idx={wargs.block_idx}, model has {len(model.blocks)} blocks")

    # Target resize for model input
    target_wh = (int(opt_args.img_size[0]), int(opt_args.img_size[1]))  # (W,H)

    orig_imgs: List[Image.Image] = []
    attn_maps: List[Optional[np.ndarray]] = []
    mamba_maps: List[Optional[np.ndarray]] = []

    for name in img_names:
        img_path = lines_dir / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        pil_orig, x_img = _load_images(img_path, target_wh)
        x_img = x_img.to(device)

        with torch.no_grad():
            x_seq = _extract_tokens_before_blocks(model, x_img)  # [1,T',C]
            x_in = _forward_to_block_input(model, x_seq, block_idx)

        block = model.blocks[block_idx]
        attn = _compute_attention_map_from_block(block, x_in)
        mamba = _compute_mamba_dependency_map_from_block(block, x_in, eps=float(wargs.eps))

        orig_imgs.append(pil_orig)
        attn_maps.append(attn)
        mamba_maps.append(mamba)

    vis_root = (REPO_ROOT / wargs.vis_dir).resolve()
    out_dir = vis_root / f"{dataset}_{opt_args.exp_name}" / f"block{block_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stem = out_dir / f"{dataset}_{opt_args.exp_name}_block{block_idx}_2imgs_attn_mamba"
    
    saved_paths = _plot_grid_3x1(
    out_stem=out_stem,
    dataset_tag=dataset,
    img_name=img_names[0],
    pil_img=orig_imgs[0],
    attn_map=attn_maps[0],
    dep_map=mamba_maps[0],
    save_formats=save_formats,
)

    print("[DONE] Saved files:")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()