"""
experiment.py — Linear vs cosine schedule comparison runner

Trains both the linear and cosine schedules under identical settings, then
generates sample grids for each.  Running this script produces everything
needed to compare the two schedules side-by-side.

Usage:
    python experiment.py                    # train + sample both schedules
    python experiment.py --epochs 10        # override epoch count for both
    python experiment.py --sample_only      # skip training, regenerate samples

All other hyperparameters are kept fixed so the only experimental variable
is the noise schedule.  Do not change anything else when doing comparisons.
"""

import argparse
import os
import subprocess
import sys


# ===========================================================================
# Shared settings — identical for both schedule runs
# ===========================================================================

SHARED = dict(
    run_name      = "run_01",
    epochs        = 100,
    timesteps     = 1000,
    batch_size    = 128,
    learning_rate = 2e-4,
    save_every    = 10,
    seed          = 42,     # same seed → same weight init, same data order
)

SCHEDULES = ["linear", "cosine"]
DATASETS  = ["fashionmnist", "cifar10"]

DATASET_CONFIG = {
    "fashionmnist": {"epochs": 100, "base_channels": 32},
    "cifar10":      {"epochs": 150, "base_channels": 64},
}


# ===========================================================================
# Helpers
# ===========================================================================

def _python() -> str:
    """Return the path to the current Python interpreter."""
    return sys.executable


def _ckpt_path(dataset: str, schedule: str, run_name: str, epochs: int, experiment_root: str) -> str:
    """Return the path to the final checkpoint for a given run."""
    return os.path.join(
        experiment_root, dataset, schedule, run_name,
        "checkpoints", f"ckpt_epoch{epochs:04d}.pt",
    )


def run_training(schedule: str, dataset: str, cfg: dict, experiment_root: str) -> None:
    """Launch train.py for one schedule as a subprocess."""
    print(f"\n{'='*60}")
    print(f"  Training: {dataset} / {schedule} schedule")
    print(f"{'='*60}\n")

    cmd = [
        _python(), "train.py",
        "--schedule",       schedule,
        "--dataset",        dataset,
        "--run_name",       cfg["run_name"],
        "--epochs",         str(cfg["epochs"]),
        "--base_channels",  str(cfg["base_channels"]),
        "--timesteps",      str(cfg["timesteps"]),
        "--batch_size",     str(cfg["batch_size"]),
        "--learning_rate",  str(cfg["learning_rate"]),
        "--save_every",     str(cfg["save_every"]),
        "--seed",           str(cfg["seed"]),
        "--experiment_root", experiment_root,
    ]
    if cfg.get("use_attention"):
        cmd.append("--use_attention")
    subprocess.run(cmd, check=True)


def run_sampling(schedule: str, dataset: str, cfg: dict, experiment_root: str) -> None:
    """Launch sample.py for a completed run."""
    ckpt = _ckpt_path(dataset, schedule, cfg["run_name"], cfg["epochs"], experiment_root)
    if not os.path.isfile(ckpt):
        print(f"  [skip] Checkpoint not found: {ckpt}")
        return

    print(f"\n{'='*60}")
    print(f"  Sampling: {schedule} schedule")
    print(f"{'='*60}\n")

    cmd = [_python(), "sample.py", "--ckpt", ckpt]
    subprocess.run(cmd, check=True)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run linear vs cosine schedule comparison"
    )
    parser.add_argument("--epochs",      type=int,   default=SHARED["epochs"])
    parser.add_argument("--timesteps",   type=int,   default=SHARED["timesteps"])
    parser.add_argument("--batch_size",  type=int,   default=SHARED["batch_size"])
    parser.add_argument("--seed",        type=int,   default=SHARED["seed"])
    parser.add_argument("--run_name",                default=SHARED["run_name"])
    parser.add_argument("--experiment_root",         default="experiments")
    parser.add_argument(
        "--sample_only", action="store_true",
        help="Skip training and only generate samples from existing checkpoints",
    )
    parser.add_argument(
        "--schedules", nargs="+", default=SCHEDULES,
        choices=SCHEDULES,
        help="Which schedules to run (default: both)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        choices=DATASETS,
        help="Which datasets to run (default: both)",
    )
    parser.add_argument(
        "--use_attention", action="store_true",
        help="Add self-attention at the bottleneck of the U-Net",
    )
    args = parser.parse_args()

    cfg = SHARED.copy()
    cfg["epochs"]     = args.epochs
    cfg["timesteps"]  = args.timesteps
    cfg["batch_size"] = args.batch_size
    cfg["seed"]       = args.seed
    cfg["run_name"]      = args.run_name
    cfg["use_attention"] = args.use_attention

    print("Linear vs cosine schedule comparison")
    print(f"  Schedules : {args.schedules}")
    print(f"  Run name  : {cfg['run_name']}")
    print(f"  Epochs    : {cfg['epochs']}")
    print(f"  Timesteps : {cfg['timesteps']}")
    print(f"  Seed      : {cfg['seed']}")
    print(f"  Root      : {args.experiment_root}")

    for dataset in args.datasets:
        ds_cfg = {**cfg, **DATASET_CONFIG[dataset]}
        if args.epochs != SHARED["epochs"]:
            ds_cfg["epochs"] = args.epochs
        for schedule in args.schedules:
            if not args.sample_only:
                run_training(schedule, dataset, ds_cfg, args.experiment_root)
            run_sampling(schedule, dataset, ds_cfg, args.experiment_root)

    print("\nAll done.")
    print(f"Results are in: {args.experiment_root}/")
    print("Run analyze.py to load the loss.csv files and sample grids from each run.")


if __name__ == "__main__":
    main()
