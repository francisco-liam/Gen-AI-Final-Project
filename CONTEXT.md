# Project Context — Noise Schedule Geometry in Diffusion Models

This file documents the project's evolution, experimental results, current
state, and remaining work.  Read this when picking up the project on a new
machine.

---

## What this project is

A from-scratch PyTorch implementation of **DDPM** (Ho et al., 2020) that
empirically compares the **linear** and **cosine** noise schedules on both
FashionMNIST and CIFAR-10.  The scientific question:

> *Does the cosine schedule (Nichol & Dhariwal, 2021) produce better sample
> quality than the linear schedule, and under what architectural conditions
> does its advantage emerge?*

The answer, as the experiments so far show, depends heavily on model capacity
— specifically the presence of self-attention at the bottleneck.

---

## Experimental history

### Phase 1 — FashionMNIST baseline (completed, attention=False)

**Setup:** 100 epochs, T=1000, base\_channels=32, no attention  
**Results:**

| Schedule | FID ↓ | IS ↑ |
|---|---|---|
| Linear | 33.93 | 3.790 |
| Cosine | 66.08 | 3.261 |

**Interpretation:** Linear wins on FashionMNIST without attention. The dataset
is 28×28 grayscale with simple, low-frequency structure — linear converges
faster because its high-noise-level gradients are more informative for simple
textures.

### Phase 2 — CIFAR-10 (completed, attention=False)

**Setup:** 150 epochs, T=1000, base\_channels=64, no attention  
**Results:**

| Schedule | FID ↓ | IS ↑ |
|---|---|---|
| Linear | 86.83 | 5.163 |
| Cosine | 191.69 | 4.216 |

**Interpretation:** Cosine is dramatically worse without attention. Cosine's
uniform SNR forces the model to reconstruct global structure from near-pure-noise
inputs (high-t). Without attention, 3×3 convolutions cannot capture long-range
spatial dependencies. The original Improved DDPM paper used multi-head
self-attention at multiple resolutions.

### Phase 3 — Self-attention implementation + FashionMNIST re-run (completed)

A `SelfAttention` module was added to `model.py` and wired through the full
pipeline behind a `--use_attention` flag. Key details:

- **Where:** Bottleneck only (7×7 for FashionMNIST, 8×8 for CIFAR-10)
- **Architecture:** GroupNorm pre-norm → 1×1 conv Q/K/V → scaled dot-product
  attention (4 heads) → residual add
- **Cost:** ~263K extra parameters for CIFAR-10 at base\_channels=64

**FashionMNIST with attention results** (100 epochs, base\_channels=32):

| Schedule | FID ↓ | IS ↑ |
|---|---|---|
| Linear | 42.34 | 4.034 |
| Cosine | **30.00** | 3.934 |

**Cosine now wins by ~30% FID** — self-attention allows the model to leverage
cosine's uniform SNR distribution. Cosine's training loss (~0.066) is ~2× the
linear loss (~0.036), which is expected: cosine assigns harder denoising tasks
at all timesteps. Gradient norms are stable for both (linear ~0.065, cosine
~0.11). All outputs archived under `fashionmnist/`.

### Phase 4 — CIFAR-10 with attention (linear complete, cosine inconclusive)

**Linear:** 150 epochs, base\_channels=64, attention=True — **converged well**,
producing visually recognizable CIFAR images. Final loss ~0.031.

**Cosine:** 150 epochs, base\_channels=64, attention=True — **did not converge
sufficiently**. Final loss ~0.057 (~2× linear). Samples are dark and blurry.
Root cause: cosine on CIFAR-10 requires either more epochs (plateau not yet
reached) or a larger model (base\_channels ≥ 96) to master the fine-grained
low-noise denoising steps that produce crisp natural images. The linear
schedule is more forgiving of limited capacity because it front-loads noise,
making medium-noise steps easier to learn.

**Decision:** Primary results use FashionMNIST (clean story, both schedules
converged). CIFAR linear results are shown as a cross-dataset generalization
check. CIFAR cosine failure is noted as a capacity/budget limitation.

---

## Current code state

| File | Status | Notes |
|---|---|---|
| `model.py` | Done | `SelfAttention` class + `use_attention` flag in `SmallUNet` |
| `train.py` | Done | `--use_attention` + `--resume` CLI; config saved in checkpoint |
| `sample.py` | Done | reads dataset/attention from checkpoint; correct image shape for CIFAR |
| `analyze.py` | Done | tqdm progress bars for FID/IS generation; per-dataset output dirs |
| `run_all.py` | Done | `--use_attention` CLI, passes to train subprocess |
| `experiment.py` | Done | `--use_attention` CLI, passes to train subprocess |
| `data.py` | Done | NumPy 2.4 DeprecationWarning suppressed for CIFAR pickle load |
| `schedule.py` | Done | unchanged |
| `diffusion.py` | Done | `sample()` and `sample_with_trajectory()` accept `image_shape` arg |

All files are Python 3.10-compatible and have no external dependencies beyond
what is listed in `requirements.txt`.

---

## Architecture summary

```
SmallUNet (base_channels=C):

  Input (B, in_ch, H, W)
    └─ enc1:  ResBlock   in_ch → C,   H×W
    └─ down1: MaxPool2d  C  → C,    H/2×W/2
    └─ enc2:  ResBlock   C  → 2C,   H/2×W/2
    └─ down2: MaxPool2d  2C → 2C,   H/4×W/4
    └─ bottleneck: ResBlock  2C → 4C,  H/4×W/4
    └─ [attn]: SelfAttention 4C heads=4   ← only if --use_attention
    └─ up1:   Upsample + ResBlock  4C+2C → 2C
    └─ up2:   Upsample + ResBlock  2C+C  → C
    └─ out:   Conv2d 1×1  C → in_ch
  Output (B, in_ch, H, W)

Timestep embedding: sinusoidal → Linear → SiLU → Linear → 4C dim
Injected via affine shift into each ResBlock.
```

For FashionMNIST (28×28): bottleneck is 7×7  
For CIFAR-10 (32×32): bottleneck is 8×8

---

## Dataset-specific hyperparameters

These are the defaults used by `run_all.py` and `experiment.py`:

```python
DATASET_CONFIG = {
    "fashionmnist": {"epochs": 100, "base_channels": 32},
    "cifar10":      {"epochs": 150, "base_channels": 64},
}
SHARED = {
    "timesteps":     1000,
    "batch_size":    128,
    "learning_rate": 2e-4,
    "save_every":    2,
    "seed":          42,
    "run_name":      "run_01",
}
```

---

## What to do when you pick this up

1. **Set up the environment** — follow [README.md](README.md) setup section.

2. **Primary results are done** — FashionMNIST with attention (both schedules).
   All outputs are in `fashionmnist/outputs/`.

3. **Re-run analysis only (no retraining needed):**
   ```bash
   # FashionMNIST
   python analyze.py \
     --experiment_root fashionmnist/experiments \
     --output_dir fashionmnist/outputs \
     --dataset fashionmnist --schedules linear cosine --run_name run_01

   # CIFAR linear only
   python analyze.py \
     --experiment_root experiments \
     --output_dir outputs/cifar10 \
     --dataset cifar10 --schedules linear --run_name run_01
   ```

4. **If a training run crashes, resume it:**
   ```bash
   python train.py --schedule cosine --dataset cifar10 --run_name run_01 \
     --epochs 150 --base_channels 64 --use_attention --resume
   ```

5. **Write the report** — key narrative:
   - Section 1: DDPM background, forward/reverse process, noise schedules
   - Section 2: SNR geometry — cosine has more uniform SNR, linear front-loads noise
   - Section 3: FashionMNIST without attention — linear wins (simple dataset)
   - Section 4: FashionMNIST with attention — cosine wins (attention unlocks cosine's advantage)
   - Section 5: CIFAR-10 — linear generalizes well; cosine requires larger model/more epochs
   - Conclusion: Architecture and dataset complexity jointly determine which schedule wins

---

## File locations

```
/home/ecsl/Gen-AI-Final-Project/
  ├── data.py, schedule.py, diffusion.py, model.py
  ├── train.py, sample.py, analyze.py
  ├── experiment.py, run_all.py
  ├── requirements.txt
  ├── README.md, CONTEXT.md
  ├── .venv/                    (virtualenv, not in git)
  ├── data/                     (downloaded datasets, not in git)
  ├── fashionmnist/             (primary results, not in git)
  │   ├── experiments/
  │   │   ├── linear/run_01/    ← 100 epochs, attention=True
  │   │   └── cosine/run_01/    ← 100 epochs, attention=True
  │   └── outputs/              ← all figures + summary_metrics.csv
  ├── experiments/              (CIFAR results, not in git)
  │   └── cifar10/
  │       ├── linear/run_01/    ← 150 epochs, attention=True  (good)
  │       └── cosine/run_01/    ← 150 epochs, attention=True  (underconverged)
  └── outputs/
      └── cifar10/              ← CIFAR figures
```

---

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
