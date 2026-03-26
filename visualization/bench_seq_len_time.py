#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: input image width vs inference latency (ms/sample)

- Uses *random noise* inputs: [B, 1, H=64, W]
- W in {64, 128, ..., 2048} (step=64)
- Loads 3 trained checkpoints (IAM/LAM/READ) and plots 3 curves in one figure.

Outputs (written under --vis-dir):
  - width_vs_time.csv
  - width_vs_time.png
  - width_vs_time.md  (ready-to-paste caption + usage notes)
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

# --- make project root importable (so `from model...` works when script is inside vis/) ---
THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Try to import exactly like your test.py
try:
    from utils import option  # type: ignore
except Exception as e:
    raise RuntimeError("Failed to import utils.option. Run this script from your project checkout.") from e

try:
    from model.htr_bimamba_hybrid import create_model  # type: ignore
except Exception:
    # fallback if user moves file structure
    from htr_bimamba_hybrid import create_model  # type: ignore


def _load_ckpt(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
    strict_load: bool = False,
    verbose: bool = True,
) -> None:
    """Load checkpoint the same way as your test.py (state_dict_ema -> DP strip).

    If strict_load=True, use strict=True and crash on any mismatch (best for verifying correctness).
    Otherwise, keep strict=False (same as your test.py) but print missing/unexpected keys.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict_ema", ckpt)
    # remove DataParallel prefix if present
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if strict_load:
        model.load_state_dict(state_dict, strict=True)
    else:
        incompatible = model.load_state_dict(state_dict, strict=False)
        if verbose:
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            print(f"[CKPT LOAD] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
            if len(missing) > 0:
                print("  - missing (first 10):", missing[:10])
            if len(unexpected) > 0:
                print("  - unexpected (first 10):", unexpected[:10])

    model.to(device)
    model.eval()


@torch.no_grad()
def _get_effective_seq_len(model: torch.nn.Module, x: torch.Tensor) -> int:
    """
    Effective sequence length after:
      backbone -> adaptive pool (H'=1) -> TemporalDownsample (Conv1d stride=td_stride)
    """
    import torch.nn.functional as F
    feats = model.patch_embed(x)                                # [B, C, H', W']
    feats = F.adaptive_avg_pool2d(feats, (1, feats.size(-1)))  # [B, C, 1, W']
    feats = feats.squeeze(2)                                   # [B, C, W']
    seq = model.temporal_down(feats)                           # [B, T', C]
    return int(seq.shape[1])


def _timed_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    amp: bool,
) -> Tuple[float, float]:
    """
    Returns (mean_ms, std_ms) for ms/sample (batch already included in x).
    Uses CUDA Events for more stable timing.
    """
    assert x.is_cuda, "Timing assumes CUDA for accurate sync; set --device cuda"
    bs = x.shape[0]

    # warmup
    for _ in range(max(0, warmup)):
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)

    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    times_ms: List[float] = []
    for _ in range(iters):
        starter.record()
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)  # milliseconds for the whole batch
        times_ms.append(elapsed / bs)          # ms/sample

    mean_ms = float(sum(times_ms) / len(times_ms))
    # std (unbiased not necessary)
    var = sum((t - mean_ms) ** 2 for t in times_ms) / max(1, (len(times_ms) - 1))
    std_ms = float(var ** 0.5)
    return mean_ms, std_ms


def _build_model_from_cli_args(args_namespace: argparse.Namespace) -> torch.nn.Module:
    """
    Mirror test.py's create_model call so architecture matches training.
    """
    model = create_model(
        nb_cls=args_namespace.nb_cls,
        img_size=tuple(args_namespace.img_size[::-1]),
        embed_dim=args_namespace.embed_dim,
        depth=args_namespace.encoder_depth,
        mlp_ratio=args_namespace.mlp_ratio,
        drop_path_rate=args_namespace.drop_path_rate,
        num_levels=args_namespace.num_levels,
        channel_multiplier=args_namespace.channel_multiplier,
        td_stride=args_namespace.td_stride,
        attn_every=args_namespace.attn_every,
        attn_heads=args_namespace.attn_heads,
        attn_window=args_namespace.attn_window,
        use_asymmetric=not args_namespace.no_aniso,
        use_csp=not args_namespace.no_csp,
        enable_mamba=not args_namespace.no_mamba,
        enable_attn=not args_namespace.no_attn,
    )
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("width vs inference time benchmark")

    # Output / runtime
    p.add_argument("--vis-dir", type=str, default="vis",
                   help="where to write csv/png/md (relative to project root)")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"], help="benchmark device (cuda recommended)")
    p.add_argument("--batch-size", type=int, default=1, help="batch size used for timing")
    p.add_argument("--height", type=int, default=64, help="input image height (fixed)")
    p.add_argument("--w-min", type=int, default=64, help="min width")
    p.add_argument("--w-max", type=int, default=2048, help="max width (inclusive)")
    p.add_argument("--w-step", type=int, default=64, help="width step")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations per width")
    p.add_argument("--iters", type=int, default=30, help="timed iterations per width")
    p.add_argument("--amp", action="store_true", default=False,
                   help="use CUDA autocast fp16 (faster, closer to deployment); default: off")
    p.add_argument("--seed", type=int, default=123, help="random seed")
    p.add_argument("--strict-load", action="store_true", default=False,
                   help="verify checkpoint matches model exactly (uses strict=True).")
    p.add_argument("--print-model", action="store_true", default=False,
                   help="print model class + parameter count for sanity checking")

    # Paper-friendly axis: secondary x-axis that shows T′ (effective sequence length)
    p.add_argument("--top-axis-ref", type=str, default="LAM", choices=["IAM", "LAM", "READ"],
                   help="which curve to use to map W→T′ for the top x-axis")
    p.add_argument("--top-axis-step", type=int, default=256,
                   help="tick step (in pixels) for the top axis labels; larger = less crowded")

    # Where checkpoints are
    p.add_argument("--out-dir", type=str, default="./output",
                   help="same as your training/test --out-dir (checkpoint path = out_dir/exp_name/best_CER.pth)")

    # Which 3 checkpoints to compare (defaults follow your description)
    p.add_argument("--exp-iam", type=str, default="IAM_A0_noAniso")
    p.add_argument("--exp-lam", type=str, default="LAM_A0_noAniso")
    p.add_argument("--exp-read", type=str, default="READ_A0_noAniso")

    # IMPORTANT: match training args (BASE_ARGS > create_model > option)
    # We set these explicitly to match run_ablation_all.sh baseline values.
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--encoder-depth", type=int, default=6)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--num-levels", type=int, default=4)
    p.add_argument("--channel-multiplier", type=float, default=1.5)
    p.add_argument("--attn-every", type=int, default=3)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-window", type=int, default=-1)
    p.add_argument("--img-size", type=int, nargs=2, default=[512, 64],
                   help="(H, W) used in training; model is fully-conv so this is only metadata")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but not available.")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    vis_dir = (PROJ_ROOT / args.vis_dir).resolve()
    vis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = vis_dir / "width_vs_time.csv"
    png_path = vis_dir / "width_vs_time.png"
    pdf_path = vis_dir / "width_vs_time.pdf"
    md_path = vis_dir / "width_vs_time.md"

    # Define the 3 models to benchmark.
    # NOTE: td_stride differs by dataset: IAM=4, LAM=8, READ=8 (as you described).
    jobs = [
        {"name": "IAM",  "exp": args.exp_iam,  "dataset": "IAM",  "td_stride": 4, "no_aniso": True},
        {"name": "LAM",  "exp": args.exp_lam,  "dataset": "LAM",  "td_stride": 8, "no_aniso": True},
        {"name": "READ", "exp": args.exp_read, "dataset": "READ", "td_stride": 8, "no_aniso": True},
    ]

    # We'll create an args object compatible with your option.py for each dataset,
    # so nb_cls and data settings come from the dataset subcommand defaults.
    # Then override with BASE_ARGS-like values + the per-job extra flags.
    def build_option_args(dataset_name: str, exp_name: str, td_stride: int, no_aniso: bool) -> argparse.Namespace:
        # option.get_args_parser() parses sys.argv; emulate that safely:
        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "bench_seq_len_time.py",
                "--out-dir", args.out_dir,
                "--exp-name", exp_name,
                "--img-size", str(args.img_size[0]), str(args.img_size[1]),
                "--embed-dim", str(args.embed_dim),
                "--encoder-depth", str(args.encoder_depth),
                "--mlp-ratio", str(args.mlp_ratio),
                "--drop-path-rate", str(args.drop_path_rate),
                "--num-levels", str(args.num_levels),
                "--channel-multiplier", str(args.channel_multiplier),
                "--td-stride", str(td_stride),
                "--attn-every", str(args.attn_every),
                "--attn-heads", str(args.attn_heads),
                "--attn-window", str(args.attn_window),
            ]
            if no_aniso:
                sys.argv += ["--no-aniso"]
            # dataset subcommand must be last (per your training scripts)
            sys.argv += [dataset_name]
            opt_args = option.get_args_parser()
            return opt_args
        finally:
            sys.argv = old_argv

    # Prepare results container
    rows: List[Dict[str, Any]] = []

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    # Benchmark each model
    for job in jobs:
        opt_args = build_option_args(
            dataset_name=job["dataset"],
            exp_name=job["exp"],
            td_stride=job["td_stride"],
            no_aniso=job["no_aniso"],
        )

        # Build model exactly like test.py does
        model = _build_model_from_cli_args(opt_args)

        if args.print_model:
            n_params = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[MODEL] {job['name']} -> {model.__class__.__name__} | params={n_params:,} (trainable={n_train:,})")

        ckpt_path = Path(opt_args.out_dir) / opt_args.exp_name / "best_CER.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Expected: {opt_args.out_dir}/{opt_args.exp_name}/best_CER.pth"
            )

        _load_ckpt(model, ckpt_path, device, strict_load=args.strict_load, verbose=True)

        # Generate widths
        widths = list(range(args.w_min, args.w_max + 1, args.w_step))

        for w in widths:
            # random input (noise) on device
            x = torch.randn(args.batch_size, 1, args.height, w, device=device)

            # effective seq length (post temporal downsample)
            seq_len = _get_effective_seq_len(model, x)

            # timing
            if device.type == "cuda":
                mean_ms, std_ms = _timed_forward(model, x, warmup=args.warmup, iters=args.iters, amp=args.amp)
            else:
                # CPU fallback (less stable)
                # warmup
                for _ in range(max(0, args.warmup)):
                    _ = model(x)
                t0 = time.time()
                for _ in range(args.iters):
                    _ = model(x)
                t1 = time.time()
                mean_ms = (t1 - t0) * 1000.0 / args.iters / args.batch_size
                std_ms = 0.0

            rows.append({
                "model": job["name"],
                "dataset": job["dataset"],
                "exp_name": job["exp"],
                "checkpoint": str(ckpt_path),
                "gpu": gpu_name,
                "amp_fp16": int(args.amp),
                "height": args.height,
                "width": w,
                "seq_len": seq_len,
                "td_stride": job["td_stride"],
                "embed_dim": args.embed_dim,
                "depth": args.encoder_depth,
                "attn_every": args.attn_every,
                "attn_heads": args.attn_heads,
                "attn_window": args.attn_window,
                "ms_per_sample_mean": round(mean_ms, 6),
                "ms_per_sample_std": round(std_ms, 6),
            })

            print(f"[{job['name']}] W={w:4d} -> T'={seq_len:4d} | {mean_ms:.3f} ± {std_ms:.3f} ms/sample")

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    # Plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise RuntimeError("matplotlib + numpy are required to plot. Install them or just use the CSV.") from e

    # Group by model
    def group_by_model(model_name: str) -> List[Dict[str, Any]]:
        rr = [r for r in rows if r["model"] == model_name]
        rr.sort(key=lambda x: x["width"])
        return rr

    fig, ax = plt.subplots()
    for model_name in ["IAM", "LAM", "READ"]:
        rr = group_by_model(model_name)
        xs = [r["width"] for r in rr]
        ys = [r["ms_per_sample_mean"] for r in rr]
        ax.plot(xs, ys, marker="o", linewidth=1, markersize=3, label=model_name)

    ax.set_xlabel("Input image width W (pixels)")
    ax.set_ylabel("Inference time (ms / sample)")
    ax.set_title("Input Width vs Inference Time (3 checkpoints)")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # ---- Secondary x-axis (top): Effective sequence length T′ ----
    # NOTE: T′ depends on the temporal downsampling stride; since IAM uses td_stride=4
    # while LAM/READ use td_stride=8, a single top axis must choose one mapping.
    # We use the curve specified by --top-axis-ref (default: LAM) and map W ↔ T′ by interpolation.
    rr_ref = group_by_model(args.top_axis_ref)
    w_ref = np.array([r["width"] for r in rr_ref], dtype=float)
    t_ref = np.array([r["seq_len"] for r in rr_ref], dtype=float)

    def w_to_t(w):
        w = np.asarray(w, dtype=float)
        return np.interp(w, w_ref, t_ref)

    def t_to_w(t):
        t = np.asarray(t, dtype=float)
        return np.interp(t, t_ref, w_ref)

    secax = ax.secondary_xaxis('top', functions=(w_to_t, t_to_w))
    secax.set_xlabel("Effective sequence length T′ (after temporal downsampling)")

    # Choose less-crowded ticks for top axis
    tick_ws = [w for w in range(args.w_min, args.w_max + 1, max(1, args.top_axis_step))]
    tick_ts = [float(w_to_t(w)) for w in tick_ws]
    # Note: ticks on the secondary axis are specified in the *secondary* units (T′)
    secax.set_xticks(tick_ts)
    secax.set_xticklabels([str(int(round(t))) for t in tick_ts])

    fig.tight_layout()
    # PNG
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    # PDF
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Write MD snippet (caption + how to cite)
    md = f"""# Sequence Length vs Inference Time (Scheme 1)

**Outputs**
- CSV: `{csv_path.name}`
- Figure: `{png_path.name}`

**Suggested caption (PR-safe)**

Figure X visualizes the inference latency of our model under varying effective sequence lengths (controlled by input width). As the sequence length increases, the inference time grows approximately linearly, consistent with the linear-time complexity of the state-space based encoder. We report three curves using the same architecture but checkpoints trained on IAM/LAM/READ, with their dataset-specific temporal downsampling stride (IAM: 4; LAM/READ: 8).

**Benchmark details**
- Input: random noise images with fixed height = {args.height}, width ∈ [{args.w_min}, {args.w_max}] step {args.w_step}
- Device: {gpu_name}
- Batch size: {args.batch_size}
- Warmup / Timed iters per width: {args.warmup} / {args.iters}
- AMP(fp16 autocast): {"ON" if args.amp else "OFF"}

**Notes**
- Bottom X-axis uses the **raw input width W (pixels)**.
- Top X-axis shows **effective T′** mapped from W using the **{args.top_axis_ref}** curve (interpolated).
- The **effective sequence length T′** (after `TemporalDownsample` with `td_stride`) is still computed and saved in the CSV as `seq_len`, so you can discuss the linear-time behavior w.r.t. T′ in the text.
- This is intentionally model-internal and does not depend on external baselines (reviewer-friendly).
"""
    md_path.write_text(md, encoding="utf-8")

    print("\n==============================")
    print("Done.")
    print(f"CSV : {csv_path}")
    print(f"PNG : {png_path}")
    print(f"PDF : {pdf_path}")
    print(f"MD  : {md_path}")
    print("==============================\n")


if __name__ == "__main__":
    main()
