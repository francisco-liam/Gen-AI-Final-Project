"""
experiment.py — Week 3 comparison runner

Trains both the linear and cosine schedules under identical settings, then
generates sample grids for each.  Running this script produces everything
needed to compare the two schedules side-by-side in Week 4.

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
    epochs        = 10,
    timesteps     = 200,
    batch_size    = 128,
    learning_rate = 2e-4,
    save_every    = 2,
    seed          = 42,     # same seed → same weight init, same data order
)

SCHEDULES = ["linear", "cosine"]


# ===========================================================================
# Helpers
# ===========================================================================

def _python() -> str:
    """Return the path to the current Python interpreter."""
    return sys.executable


def _ckpt_path(schedule: str, run_name: str, epochs: int, experiment_root: str) -> str:
    """Return the path to the final checkpoint for a given run."""
    return os.path.join(
        experiment_root, schedule, run_name,
        "checkpoints", f"ckpt_epoch{epochs:04d}.pt",
    )


def run_training(schedule: str, cfg: dict, experiment_root: str) -> None:
    """Launch train.py for one schedule as a subprocess."""
    print(f"\n{'='*60}")
    print(f"  Training: {schedule} schedule")
    print(f"{'='*60}\n")

    cmd = [
        _python(), "train.py",
        "--schedule",       schedule,
        "--run_name",       cfg["run_name"],
        "--epochs",         str(cfg["epochs"]),
        "--timesteps",      str(cfg["timesteps"]),
        "--batch_size",     str(cfg["batch_size"]),
        "--learning_rate",  str(cfg["learning_rate"]),
        "--save_every",     str(cfg["save_every"]),
        "--seed",           str(cfg["seed"]),
        "--experiment_root", experiment_root,
    ]
    subprocess.run(cmd, check=True)


def run_sampling(schedule: str, cfg: dict, experiment_root: str) -> None:
    """Launch sample.py for a completed run."""
    ckpt = _ckpt_path(schedule, cfg["run_name"], cfg["epochs"], experiment_root)
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
        description="Run linear vs cosine schedule comparison (Week 3)"
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
    args = parser.parse_args()

    cfg = SHARED.copy()
    cfg["epochs"]     = args.epochs
    cfg["timesteps"]  = args.timesteps
    cfg["batch_size"] = args.batch_size
    cfg["seed"]       = args.seed
    cfg["run_name"]   = args.run_name

    print("Week 3 comparison experiment")
    print(f"  Schedules : {args.schedules}")
    print(f"  Run name  : {cfg['run_name']}")
    print(f"  Epochs    : {cfg['epochs']}")
    print(f"  Timesteps : {cfg['timesteps']}")
    print(f"  Seed      : {cfg['seed']}")
    print(f"  Root      : {args.experiment_root}")

    for schedule in args.schedules:
        if not args.sample_only:
            run_training(schedule, cfg, args.experiment_root)
        run_sampling(schedule, cfg, args.experiment_root)

    print("\nAll done.")
    print(f"Results are in: {args.experiment_root}/")
    print("Week 4 analysis can load the loss.csv files and sample grids from each run.")


if __name__ == "__main__":
    main()
