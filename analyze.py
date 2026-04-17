"""
analyze.py — Analysis pipeline

Loads experiment outputs produced by train.py and sample.py and
generates polished figures and a summary table for use in a report or slides.

Outputs (all saved to outputs/):
    snr_comparison.png        — SNR(t) curves for linear vs cosine
    loss_comparison.png       — training loss curves
    loss_by_t_comparison.png  — mean MSE per timestep bucket
    gradnorm_comparison.png   — gradient norm curves (skipped if logs absent)
    sample_comparison.png     — side-by-side generated sample grids
    fid_is_comparison.png     — FID / Inception Score bar charts (optional)
    summary_metrics.csv       — summary statistics table

Usage:
    python analyze.py
    python analyze.py --experiment_root experiments --output_dir outputs
    python analyze.py --schedules linear cosine --run_name run_01
    python analyze.py --fid_samples 1000   # generate N samples for FID/IS
    python analyze.py --skip_fid           # skip FID/IS computation
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PIL import
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("Pillow not found. sample_comparison.png will be skipped.", stacklevel=1)

# ---------------------------------------------------------------------------
# Optional torchmetrics import — needed for FID / IS
# ---------------------------------------------------------------------------
try:
    import torch
    import torchvision
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False


# ===========================================================================
# Matplotlib style
# ===========================================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "figure.constrained_layout.use": True,
})

COLORS     = {"linear": "#2196F3", "cosine": "#E91E63"}
LINESTYLES = {"linear": "-",       "cosine": "--"}


# ===========================================================================
# A.  Experiment loader
# ===========================================================================

def find_run_dir(experiment_root, schedule, run_name, dataset="fashionmnist"):
    for candidate in [
        experiment_root / dataset / schedule / run_name,
        experiment_root / dataset / schedule,
        experiment_root / schedule / run_name,
        experiment_root / schedule,
    ]:
        if (candidate / "config.json").exists():
            return candidate
    return None


def load_experiment(experiment_root, schedule, run_name, dataset="fashionmnist"):
    run_dir = find_run_dir(experiment_root, schedule, run_name, dataset)
    if run_dir is None:
        print(f"[WARNING] Run directory not found for schedule='{schedule}' "
              f"run='{run_name}' under '{experiment_root}'. Skipping.")
        return None

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    def try_csv(paths, label):
        for p in paths:
            if p.exists():
                return pd.read_csv(p)
        return None

    loss_df = try_csv(
        [run_dir / "logs" / "loss.csv", run_dir / "loss.csv"],
        f"loss.csv for '{schedule}'",
    )
    if loss_df is None:
        print(f"[WARNING] No loss.csv found for schedule='{schedule}'.")

    gradnorm_df = try_csv(
        [run_dir / "logs" / "gradnorm.csv", run_dir / "gradnorm.csv"],
        f"gradnorm.csv for '{schedule}' (optional)",
    )

    loss_by_t_df = try_csv(
        [run_dir / "logs" / "loss_by_t.csv", run_dir / "loss_by_t.csv"],
        f"loss_by_t.csv for '{schedule}' (optional)",
    )

    samples_dir  = run_dir / "samples"
    sample_paths = sorted(samples_dir.glob("*.png")) if samples_dir.exists() else []
    if not sample_paths:
        print(f"[WARNING] No sample images found for schedule='{schedule}'.")

    return dict(
        schedule=schedule,
        run_dir=run_dir,
        config=config,
        loss_df=loss_df,
        gradnorm_df=gradnorm_df,
        loss_by_t_df=loss_by_t_df,
        sample_paths=sample_paths,
    )


# ===========================================================================
# B.  SNR computation (pure numpy)
# ===========================================================================

def _linear_betas(timesteps, beta_start=1e-4, beta_end=2e-2):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def _cosine_betas(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    ab = np.cos(((x / timesteps) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ab = ab / ab[0]
    return np.clip(1.0 - (ab[1:] / ab[:-1]), 0.0, 0.999)


def compute_snr(betas):
    """SNR(t) = alpha_bar_t / (1 - alpha_bar_t)"""
    ab = np.cumprod(1.0 - betas)
    return ab / (1.0 - ab)


def get_betas_from_config(config):
    schedule  = config.get("schedule", "linear")
    timesteps = int(config.get("timesteps", 200))
    if schedule == "linear":
        return _linear_betas(timesteps)
    elif schedule == "cosine":
        return _cosine_betas(timesteps)
    print(f"[WARNING] Unknown schedule '{schedule}', defaulting to linear.")
    return _linear_betas(timesteps)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def _smooth(values, window=5):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _resolve_loss_xy(df, sched):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "avg_loss" in df.columns and "epoch" in df.columns:
        return df["epoch"].values, df["avg_loss"].values
    if "loss" in df.columns and "epoch" in df.columns:
        g = df.groupby("epoch")["loss"].mean().reset_index()
        return g["epoch"].values, g["loss"].values
    if "loss" in df.columns:
        return np.arange(1, len(df) + 1), df["loss"].values
    print(f"[WARNING] Unrecognised loss CSV columns for '{sched}': {list(df.columns)}.")
    return None


# ===========================================================================
# C.  SNR comparison
# ===========================================================================

def plot_snr_comparison(experiments, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for exp in experiments:
        sched = exp["schedule"]
        snr   = compute_snr(get_betas_from_config(exp["config"]))
        t     = np.arange(1, len(snr) + 1)
        kw    = dict(color=COLORS.get(sched), linestyle=LINESTYLES.get(sched, "-"),
                     label=sched.capitalize())
        axes[0].plot(t, snr, **kw)
        axes[1].semilogy(t, snr, **kw)
    for ax, scale in zip(axes, ["Linear scale", "Log scale"]):
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("SNR(t)")
        ax.set_title(f"SNR Curve — {scale}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Signal-to-Noise Ratio: Linear vs Cosine Schedule", fontsize=14)
    _save(fig, output_dir / "snr_comparison.png")


# ===========================================================================
# D.  Training loss comparison
# ===========================================================================

def plot_loss_comparison(experiments, output_dir):
    if not any(e["loss_df"] is not None for e in experiments):
        print("[WARNING] No loss data — skipping loss_comparison.png.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for exp in experiments:
        if exp["loss_df"] is None:
            continue
        sched = exp["schedule"]
        xy    = _resolve_loss_xy(exp["loss_df"], sched)
        if xy is None:
            continue
        x, y  = xy
        color, ls = COLORS.get(sched), LINESTYLES.get(sched, "-")
        ax.plot(x, y, color=color, linestyle=ls, alpha=0.35, linewidth=1)
        ax.plot(x, _smooth(y), color=color, linestyle=ls,
                label=f"{sched.capitalize()} (smoothed)", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Training Loss: Linear vs Cosine Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "loss_comparison.png")


# ===========================================================================
# E.  Loss by timestep bucket
# ===========================================================================

def plot_loss_by_timestep(experiments, output_dir):
    """
    Plot mean MSE per timestep bucket (final epoch + mean across all epochs).

    High loss at low t  → model struggles with fine detail (low noise level).
    High loss at high t → model struggles with coarse structure (high noise).
    Reads logs/loss_by_t.csv: columns [epoch, bucket_0, bucket_1, ...].
    """
    if not any(e["loss_by_t_df"] is not None for e in experiments):
        print("[INFO] No loss_by_t.csv logs — skipping loss_by_t_comparison.png.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for exp in experiments:
        df = exp["loss_by_t_df"]
        if df is None:
            continue
        sched = exp["schedule"]
        T     = int(exp["config"].get("timesteps", 200))
        color, ls, label = COLORS.get(sched), LINESTYLES.get(sched, "-"), sched.capitalize()

        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        bucket_cols = [c for c in df.columns if c.startswith("bucket_")]
        if not bucket_cols:
            print(f"[WARNING] No bucket columns in loss_by_t.csv for '{sched}'.")
            continue

        n_buckets   = len(bucket_cols)
        bucket_mids = np.array([int((i + 0.5) * T / n_buckets) for i in range(n_buckets)])
        final_row   = df[bucket_cols].iloc[-1].astype(float).values
        epoch_mean  = df[bucket_cols].astype(float).mean(axis=0).values

        kw = dict(color=color, linestyle=ls, marker="o", markersize=4, label=label)
        axes[0].plot(bucket_mids, final_row,  **kw)
        axes[1].plot(bucket_mids, epoch_mean, **kw)

    for ax, title in zip(axes, ["Final epoch", "Mean across all epochs"]):
        ax.set_xlabel("Timestep bucket midpoint")
        ax.set_ylabel("Mean MSE")
        ax.set_title(f"Loss by Timestep — {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Timestep MSE: Linear vs Cosine Schedule", fontsize=14)
    _save(fig, output_dir / "loss_by_t_comparison.png")


# ===========================================================================
# F.  Sample comparison
# ===========================================================================

def _pick_best_sample(sample_paths):
    if not sample_paths:
        return None
    for p in sample_paths:
        if p.name == "samples_latest.png":
            return p
    epoch_files = [p for p in sample_paths if "epoch" in p.name and "trajectory" not in p.name]
    if epoch_files:
        return sorted(epoch_files)[-1]
    non_traj = [p for p in sample_paths if "trajectory" not in p.name]
    return non_traj[-1] if non_traj else sample_paths[-1]


def plot_sample_comparison(experiments, output_dir):
    if not PIL_AVAILABLE:
        print("[WARNING] Pillow unavailable — skipping sample_comparison.png.")
        return
    panels = []
    for exp in experiments:
        best = _pick_best_sample(exp["sample_paths"])
        if best is None:
            print(f"[WARNING] No sample image for schedule='{exp['schedule']}'.")
            continue
        panels.append((exp["schedule"].capitalize(), Image.open(best).convert("RGB")))
    if not panels:
        print("[WARNING] No sample images — skipping sample_comparison.png.")
        return
    target_h = 256
    resized = [(lbl, img.resize((int(target_h * img.width / img.height), target_h), Image.LANCZOS))
               for lbl, img in panels]
    n = len(resized)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (lbl, img) in zip(axes, resized):
        ax.imshow(np.array(img))
        ax.set_title(lbl, fontsize=13, pad=8)
        ax.axis("off")
    fig.suptitle("Generated Samples: Linear vs Cosine Schedule", fontsize=14, y=1.02)
    _save(fig, output_dir / "sample_comparison.png")


# ===========================================================================
# G.  Gradient norm comparison
# ===========================================================================

def plot_gradnorm_comparison(experiments, output_dir):
    if not any(e["gradnorm_df"] is not None for e in experiments):
        print("[INFO] No gradient norm logs — skipping gradnorm_comparison.png.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for exp in experiments:
        df = exp["gradnorm_df"]
        if df is None:
            continue
        sched = exp["schedule"]
        df    = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        if "avg_gradnorm" in df.columns and "epoch" in df.columns:
            x, y = df["epoch"].values, df["avg_gradnorm"].values
        elif "gradnorm" in df.columns and "epoch" in df.columns:
            g = df.groupby("epoch")["gradnorm"].mean().reset_index()
            x, y = g["epoch"].values, g["gradnorm"].values
        elif "gradnorm" in df.columns:
            x, y = np.arange(1, len(df) + 1), df["gradnorm"].values
        else:
            print(f"[WARNING] Unrecognised gradnorm columns for '{sched}': {list(df.columns)}.")
            continue
        color, ls = COLORS.get(sched), LINESTYLES.get(sched, "-")
        ax.plot(x, y, color=color, linestyle=ls, alpha=0.4, linewidth=1)
        ax.plot(x, _smooth(y), color=color, linestyle=ls,
                label=f"{sched.capitalize()} (smoothed)", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient L2 Norm (clipped at 1.0)")
    ax.set_title("Gradient Norm: Linear vs Cosine Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "gradnorm_comparison.png")


# ===========================================================================
# H.  FID / Inception Score
# ===========================================================================

def _load_model_and_diffusion(run_dir, config):
    """Load final checkpoint; return (model, diffusion) or (None, None)."""
    import torch
    from diffusion import GaussianDiffusion
    from model     import SmallUNet
    from schedule  import get_beta_schedule

    ckpt_dir = run_dir / "checkpoints"
    ckpts    = sorted(ckpt_dir.glob("ckpt_epoch*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        print(f"[WARNING] No checkpoints in {ckpt_dir}.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpts[-1], map_location=device)
    cfg    = ckpt.get("config", config)

    from data import DATASET_INFO
    dataset_name = cfg.get("dataset", "fashionmnist")
    in_channels  = DATASET_INFO.get(dataset_name, {"in_channels": 1})["in_channels"]

    model = SmallUNet(
        in_channels=in_channels,
        base_channels=cfg.get("base_channels", 32),
        time_dim=cfg.get("time_dim", 128),
        use_attention=cfg.get("use_attention", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    betas     = get_beta_schedule(cfg.get("schedule", "linear"), cfg.get("timesteps", 200))
    diffusion = GaussianDiffusion(betas)
    for attr in ("betas", "alphas", "alpha_bars",
                 "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars",
                 "sqrt_recip_alphas", "betas_over_sqrt_one_minus_alpha_bars",
                 "sqrt_betas"):
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    return model, diffusion


def _generate_samples_for_fid(model, diffusion, n_samples, in_channels=1, image_size=28, batch_size=64):
    """
    Run full DDPM reverse diffusion to generate n_samples images.
    Returns uint8 RGB tensor (N, 3, H, W) for torchmetrics.
    """
    import torch
    from tqdm import tqdm
    device   = next(model.parameters()).device
    T        = diffusion.T
    all_imgs = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in tqdm(range(0, n_samples, batch_size), total=n_batches, desc="Generating", unit="batch"):
            bs = min(batch_size, n_samples - start)
            x  = torch.randn(bs, in_channels, image_size, image_size, device=device)

            for t_idx in tqdm(reversed(range(T)), total=T, desc=f"  Denoising batch {start//batch_size+1}/{n_batches}", leave=False):
                t_batch  = torch.full((bs,), t_idx, device=device, dtype=torch.long)
                eps_pred = model(x, t_batch)

                # DDPM reverse step
                sqrt_recip = diffusion.sqrt_recip_alphas[t_idx].view(1, 1, 1, 1)
                beta_coeff = diffusion.betas_over_sqrt_one_minus_alpha_bars[t_idx].view(1, 1, 1, 1)
                mean       = sqrt_recip * (x - beta_coeff * eps_pred)

                if t_idx > 0:
                    noise = torch.randn_like(x)
                    x = mean + diffusion.sqrt_betas[t_idx].view(1, 1, 1, 1) * noise
                else:
                    x = mean

            # Scale [-1,1] → uint8; expand grayscale to RGB for torchmetrics
            x_uint8 = (x.clamp(-1, 1).add(1).mul(127.5)).byte()
            x_rgb   = x_uint8.expand(-1, 3, -1, -1) if in_channels == 1 else x_uint8
            all_imgs.append(x_rgb.cpu())

    return torch.cat(all_imgs, dim=0)


def _get_real_tensors(dataset_name, n_samples):
    """Load n_samples test images as uint8 RGB (N, 3, H, W) for torchmetrics."""
    import torch, torchvision, torchvision.transforms as T
    if dataset_name == "fashionmnist":
        ds = torchvision.datasets.FashionMNIST(
            root="data", train=False, download=True, transform=T.ToTensor()
        )
    else:
        ds = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True, transform=T.ToTensor()
        )
    loader   = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    imgs     = []
    for x, _ in loader:
        imgs.append(x)
        if sum(i.shape[0] for i in imgs) >= n_samples:
            break
    all_imgs = torch.cat(imgs, dim=0)[:n_samples]   # (N, C, H, W) float [0,1]
    all_uint8 = (all_imgs * 255.0).byte()
    if all_uint8.shape[1] == 1:                      # grayscale → RGB
        all_uint8 = all_uint8.expand(-1, 3, -1, -1).contiguous()
    return all_uint8


def compute_fid_is(experiments, output_dir, n_samples=1000):
    """
    Compute FID and IS for each schedule by generating n_samples images.

    FID: lower = better (compares feature distributions against real data).
    IS:  higher = better (measures diversity and sharpness of generated images).

    Note: FID with InceptionV3 on 28x28 grayscale images is approximate;
    use these numbers for relative schedule comparison, not absolute benchmarks.
    """
    if not TORCHMETRICS_AVAILABLE:
        print("[INFO] torchmetrics not available — skipping FID/IS.\n"
              "       Install with: pip install torchmetrics[image]")
        return {}

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FID/IS] Generating {n_samples} samples per schedule on {device}...")

    # Derive dataset from first experiment's config
    from data import DATASET_INFO
    first_cfg    = experiments[0]["config"]
    dataset_name = first_cfg.get("dataset", "fashionmnist")

    print(f"[FID/IS] Loading real {dataset_name} test images...")
    real_imgs = _get_real_tensors(dataset_name, n_samples).to(device)

    results = {}
    for exp in experiments:
        sched = exp["schedule"]
        print(f"[FID/IS] Generating fake images for '{sched}'...")
        model, diffusion = _load_model_and_diffusion(exp["run_dir"], exp["config"])
        if model is None:
            print(f"[WARNING] Could not load model for '{sched}' — skipping FID/IS.")
            continue

        cfg      = exp["config"]
        ds_name  = cfg.get("dataset", "fashionmnist")
        info     = DATASET_INFO.get(ds_name, {"in_channels": 1, "image_size": 28})
        fake_imgs = _generate_samples_for_fid(
            model, diffusion, n_samples,
            in_channels=info["in_channels"],
            image_size=info["image_size"],
        ).to(device)

        fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        fid_metric.update(real_imgs, real=True)
        fid_metric.update(fake_imgs, real=False)
        fid_val = fid_metric.compute().item()

        is_metric = InceptionScore(normalize=False).to(device)
        is_metric.update(fake_imgs)
        is_mean, is_std = [v.item() for v in is_metric.compute()]

        results[sched] = {"fid": fid_val, "is_mean": is_mean, "is_std": is_std}
        print(f"[FID/IS] {sched}: FID={fid_val:.2f}  IS={is_mean:.3f}±{is_std:.3f}")

        del model, diffusion, fake_imgs, fid_metric, is_metric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def plot_fid_is(fid_is_results, output_dir):
    if not fid_is_results:
        return
    schedules = list(fid_is_results.keys())
    fid_vals  = [fid_is_results[s]["fid"]     for s in schedules]
    is_means  = [fid_is_results[s]["is_mean"] for s in schedules]
    is_stds   = [fid_is_results[s]["is_std"]  for s in schedules]
    colors    = [COLORS.get(s, "gray") for s in schedules]
    labels    = [s.capitalize() for s in schedules]

    fig, (ax_fid, ax_is) = plt.subplots(1, 2, figsize=(8, 4))

    bars = ax_fid.bar(labels, fid_vals, color=colors, edgecolor="black", linewidth=0.7)
    ax_fid.set_ylabel("FID ↓")
    ax_fid.set_title("Fréchet Inception Distance")
    ax_fid.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax_fid.grid(True, axis="y", alpha=0.3)

    bars2 = ax_is.bar(labels, is_means, yerr=is_stds, color=colors,
                      edgecolor="black", linewidth=0.7, capsize=5)
    ax_is.set_ylabel("IS ↑")
    ax_is.set_title("Inception Score")
    ax_is.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
    ax_is.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Sample Quality Metrics: Linear vs Cosine Schedule", fontsize=13)
    _save(fig, output_dir / "fid_is_comparison.png")


# ===========================================================================
# I.  Summary table
# ===========================================================================

def build_summary_table(experiments, fid_is_results, output_dir):
    rows = []
    for exp in experiments:
        sched = exp["schedule"]
        cfg   = exp["config"]
        row   = {
            "Schedule":      sched.capitalize(),
            "Timesteps":     cfg.get("timesteps", "—"),
            "Epochs":        cfg.get("epochs",    "—"),
            "Learning Rate": cfg.get("learning_rate", "—"),
            "Batch Size":    cfg.get("batch_size", "—"),
        }
        if exp["loss_df"] is not None:
            xy = _resolve_loss_xy(exp["loss_df"], sched)
            if xy is not None:
                losses = xy[1]
                tail   = losses[-5:] if len(losses) >= 5 else losses
                row["Final Train Loss"]          = f"{losses[-1]:.6f}"
                row["Best Train Loss"]           = f"{losses.min():.6f}"
                row["Mean Loss (Last 5 Epochs)"] = f"{tail.mean():.6f}"
            else:
                row.update({"Final Train Loss": "—", "Best Train Loss": "—",
                            "Mean Loss (Last 5 Epochs)": "—"})
        else:
            row.update({"Final Train Loss": "—", "Best Train Loss": "—",
                        "Mean Loss (Last 5 Epochs)": "—"})
        if sched in fid_is_results:
            r = fid_is_results[sched]
            row["FID ↓"]       = f"{r['fid']:.2f}"
            row["IS ↑ (mean)"] = f"{r['is_mean']:.3f}"
            row["IS ↑ (std)"]  = f"{r['is_std']:.3f}"
        rows.append(row)

    if not rows:
        print("[WARNING] No experiments loaded — summary table empty.")
        return

    summary_df = pd.DataFrame(rows).set_index("Schedule")
    out_path   = output_dir / "summary_metrics.csv"
    summary_df.to_csv(out_path)
    print(f"[OK] Saved: {out_path}")
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(summary_df.to_string())
    print("=" * 70 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analysis pipeline: generate figures and summary table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment_root", default="experiments")
    parser.add_argument("--output_dir",      default=None,
                        help="Where to save outputs (default: outputs/<dataset>)")
    parser.add_argument("--dataset",         default="fashionmnist",
                        help="Which dataset's experiments to analyze (fashionmnist or cifar10)")
    parser.add_argument("--schedules", nargs="+", default=["linear", "cosine"])
    parser.add_argument("--run_name",  default="run_01")
    parser.add_argument("--fid_samples", type=int, default=1000,
                        help="Images to generate for FID/IS (more = more accurate, slower)")
    parser.add_argument("--skip_fid", action="store_true",
                        help="Skip FID/IS even if torchmetrics is installed")
    return parser.parse_args()


def main():
    args            = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir      = Path(args.output_dir) if args.output_dir else Path("outputs") / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading experiments from: {experiment_root.resolve()}")
    experiments = []
    for schedule in args.schedules:
        exp = load_experiment(experiment_root, schedule, args.run_name, args.dataset)
        if exp is not None:
            experiments.append(exp)
            print(f"  [loaded] {schedule}/{args.run_name}  "
                  f"(T={exp['config'].get('timesteps')}, "
                  f"epochs={exp['config'].get('epochs')})")

    if not experiments:
        print("\n[ERROR] No experiments loaded. Exiting.")
        sys.exit(1)

    print(f"\nGenerating outputs → {output_dir.resolve()}\n")

    plot_snr_comparison(experiments, output_dir)        # C
    plot_loss_comparison(experiments, output_dir)       # D
    plot_loss_by_timestep(experiments, output_dir)      # E
    plot_sample_comparison(experiments, output_dir)     # F
    plot_gradnorm_comparison(experiments, output_dir)   # G

    fid_is_results = {}
    if not args.skip_fid:
        fid_is_results = compute_fid_is(experiments, output_dir, n_samples=args.fid_samples)
        plot_fid_is(fid_is_results, output_dir)         # H
    else:
        print("[INFO] FID/IS skipped (--skip_fid).")

    build_summary_table(experiments, fid_is_results, output_dir)   # I
    print("Analysis complete.\n")


if __name__ == "__main__":
    main()
