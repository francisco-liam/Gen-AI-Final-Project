"""
run_all.py — Full pipeline: train both schedules on both datasets, sample, then analyze.

Running this will:
  1. Train  linear + cosine  on Fashion-MNIST  → experiments/fashionmnist/
  2. Train  linear + cosine  on CIFAR-10       → experiments/cifar10/
  3. Generate sample grids for all four runs
  4. Run analysis for Fashion-MNIST            → outputs/fashionmnist/
  5. Run analysis for CIFAR-10                 → outputs/cifar10/

Usage:
    python run_all.py                              # full pipeline, both datasets
    python run_all.py --datasets fashionmnist      # one dataset only
    python run_all.py --skip_train                 # skip training, re-run sampling + analysis
    python run_all.py --skip_train --skip_sample   # analysis only
    python run_all.py --skip_fid                   # skip FID/IS (faster)
    python run_all.py --epochs 2                   # quick smoke-test
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ===========================================================================
# Shared hyperparameters — identical for all schedule/dataset combinations
# ===========================================================================

SHARED = dict(
    run_name       = "run_01",
    epochs         = 100,
    timesteps      = 1000,
    batch_size     = 128,
    learning_rate  = 2e-4,
    save_every     = 10,
    seed           = 42,
    experiment_root = "experiments",
)

SCHEDULES = ["linear", "cosine"]
DATASETS  = ["fashionmnist", "cifar10"]

# Per-dataset overrides — values that differ between datasets.
# These are merged on top of SHARED when building each training command.
DATASET_CONFIG = {
    "fashionmnist": {
        "epochs":        100,
        "base_channels": 32,   # ~1M params; sufficient for 28x28 grayscale
    },
    "cifar10": {
        "epochs":        150,  # RGB + more complexity = slower convergence
        "base_channels": 64,  # ~4M params; needed for natural image textures
    },
}


# ===========================================================================
# Helpers
# ===========================================================================

def _python():
    return sys.executable


def _run(cmd: list[str], step_label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {step_label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {result.returncode}: {step_label}")
        print("Continuing with remaining steps...\n")


def _ckpt_path(dataset: str, schedule: str, cfg: dict) -> str:
    epochs = cfg["epochs"]
    return (
        f"{cfg['experiment_root']}/{dataset}/{schedule}/{cfg['run_name']}"
        f"/checkpoints/ckpt_epoch{epochs:04d}.pt"
    )


# ===========================================================================
# Pipeline steps
# ===========================================================================

def train_schedule(schedule: str, dataset: str, cfg: dict) -> None:
    cmd = [
        _python(), "train.py",
        "--schedule",        schedule,
        "--dataset",         dataset,
        "--run_name",        cfg["run_name"],
        "--epochs",          str(cfg["epochs"]),
        "--base_channels",   str(cfg["base_channels"]),
        "--timesteps",       str(cfg["timesteps"]),
        "--batch_size",      str(cfg["batch_size"]),
        "--learning_rate",   str(cfg["learning_rate"]),
        "--save_every",      str(cfg["save_every"]),
        "--seed",            str(cfg["seed"]),
        "--experiment_root", cfg["experiment_root"],
    ]
    if cfg.get("use_attention"):
        cmd.append("--use_attention")
    _run(cmd, f"Training: {dataset} / {schedule} schedule")


def sample_schedule(schedule: str, dataset: str, cfg: dict) -> None:
    ckpt = _ckpt_path(dataset, schedule, cfg)
    if not Path(ckpt).exists():
        print(f"[WARNING] Checkpoint not found, skipping sampling: {ckpt}")
        return
    cmd = [_python(), "sample.py", "--ckpt", ckpt]
    _run(cmd, f"Sampling: {dataset} / {schedule} schedule")


def run_analysis(dataset: str, cfg: dict, skip_fid: bool, fid_samples: int) -> None:
    output_dir = f"outputs/{dataset}"
    cmd = [
        _python(), "analyze.py",
        "--experiment_root", cfg["experiment_root"],
        "--output_dir",      output_dir,
        "--dataset",         dataset,
        "--schedules",       *SCHEDULES,
        "--run_name",        cfg["run_name"],
        "--fid_samples",     str(fid_samples),
    ]
    if skip_fid:
        cmd.append("--skip_fid")
    _run(cmd, f"Analysis: {dataset}")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline: train → sample → analyze (both datasets)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip_train",  action="store_true",
                        help="Skip training (use existing checkpoints)")
    parser.add_argument("--skip_sample", action="store_true",
                        help="Skip sample generation (use existing sample images)")
    parser.add_argument("--skip_fid",   action="store_true",
                        help="Skip FID/IS computation during analysis")
    parser.add_argument("--fid_samples", type=int, default=1000,
                        help="Number of images to generate for FID/IS")
    parser.add_argument("--epochs", type=int, default=SHARED["epochs"],
                        help="Override epoch count (useful for quick tests)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS,
                        help="Which datasets to run (default: both)")
    parser.add_argument("--use_attention", action="store_true",
                        help="Add self-attention at the bottleneck of the U-Net")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = SHARED.copy()
    cfg["epochs"]       = args.epochs
    cfg["use_attention"] = args.use_attention

    print("\n" + "=" * 60)
    print("  FULL PIPELINE: train → sample → analyze")
    print("=" * 60)
    print(f"  Datasets  : {args.datasets}")
    print(f"  Schedules : {SCHEDULES}")
    print(f"  Epochs    : {cfg['epochs']}")
    print(f"  Timesteps : {cfg['timesteps']}")
    print(f"  Skip train : {args.skip_train}")
    print(f"  Skip sample: {args.skip_sample}")
    print(f"  Skip FID   : {args.skip_fid}")
    print(f"  Attention  : {args.use_attention}")
    print("=" * 60)

    for dataset in args.datasets:
        print(f"\n{'#'*60}")
        print(f"  DATASET: {dataset.upper()}")
        print(f"{'#'*60}")

        # Build dataset-specific config by merging SHARED + DATASET_CONFIG overrides
        ds_cfg = {**cfg, **DATASET_CONFIG[dataset]}
        # Respect CLI --epochs override if the user explicitly passed it
        if args.epochs != SHARED["epochs"]:
            ds_cfg["epochs"] = args.epochs

        print(f"  epochs={ds_cfg['epochs']}  base_channels={ds_cfg['base_channels']}  attention={ds_cfg['use_attention']}")

        # ── Step 1: Training ─────────────────────────────────────────
        if not args.skip_train:
            for schedule in SCHEDULES:
                train_schedule(schedule, dataset, ds_cfg)
        else:
            print(f"\n[SKIP] Training skipped for {dataset}.")

        # ── Step 2: Sampling ─────────────────────────────────────────
        if not args.skip_sample:
            for schedule in SCHEDULES:
                sample_schedule(schedule, dataset, ds_cfg)
        else:
            print(f"\n[SKIP] Sampling skipped for {dataset}.")

        # ── Step 3: Analysis ─────────────────────────────────────────
        run_analysis(dataset, ds_cfg, skip_fid=args.skip_fid, fid_samples=args.fid_samples)

    print("\nAll done. Outputs:")
    for dataset in args.datasets:
        print(f"  outputs/{dataset}/")


if __name__ == "__main__":
    main()

