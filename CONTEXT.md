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

### Phase 1 — FashionMNIST baseline (completed)

**Setup:** 100 epochs, T=1000, base\_channels=32, no attention  
**Results:**

| Schedule | FID ↓ | IS ↑ |
|---|---|---|
| Linear | 33.93 | 3.790 |
| Cosine | 66.08 | 3.261 |

**Interpretation:** Linear wins on FashionMNIST.  The dataset is 28×28
grayscale with simple, low-frequency structure — linear converges faster
because its high-noise-level gradients are more informative for simple
textures.  Cosine's advantage requires datasets with fine spatial detail that
benefit from its uniform SNR distribution.

### Phase 2 — CIFAR-10 (completed, attention=False)

**Setup:** 150 epochs, T=1000, base\_channels=64, no attention  
**Results:**

| Schedule | FID ↓ | IS ↑ |
|---|---|---|
| Linear | 86.83 | 5.163 |
| Cosine | 191.69 | 4.216 |

**Interpretation:** Cosine is still dramatically worse.  Root cause diagnosed
as missing self-attention.  Cosine's uniform SNR forces the model to
reconstruct global structure from near-pure-noise inputs (high-t).  Without
attention, 3×3 convolutions cannot capture long-range spatial dependencies,
so the model fails at those timesteps.  The original Improved DDPM paper used
multi-head self-attention at multiple resolutions; our model had none.

### Phase 3 — Self-attention implementation (code done, runs pending)

A `SelfAttention` module was added to `model.py` and wired through the full
pipeline behind a `--use_attention` flag.  Key details:

- **Where:** Bottleneck only (7×7 for FashionMNIST, 8×8 for CIFAR-10)
- **Architecture:** GroupNorm pre-norm → 1×1 conv Q/K/V → scaled dot-product
  attention (4 heads) → residual add
- **Cost:** ~263K extra parameters for CIFAR-10 at base\_channels=64

**Hypothesis:** With attention, cosine should outperform linear on CIFAR-10
because the model can reconstruct global structure at high noise levels.

**These runs have not been executed yet.**  The next step is to run:

```bash
python run_all.py --datasets cifar10 --use_attention
```

---

## Current code state

| File | Status | Notes |
|---|---|---|
| `model.py` | Done | `SelfAttention` class + `use_attention` flag in `SmallUNet` |
| `train.py` | Done | `--use_attention` CLI, saved in checkpoint config |
| `sample.py` | Done | reads `use_attention` from checkpoint, no manual flag needed |
| `analyze.py` | Done | reads `use_attention` from checkpoint; per-dataset output dirs |
| `run_all.py` | Done | `--use_attention` CLI, passes to train subprocess |
| `experiment.py` | Done | `--use_attention` CLI, passes to train subprocess |
| `data.py` | Done | supports both `fashionmnist` and `cifar10` |
| `schedule.py` | Done | unchanged from original |
| `diffusion.py` | Done | unchanged from original |

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

## What to do when you pick this up tomorrow

1. **Set up the environment** — follow [README.md](README.md) setup section.

2. **Run the attention experiment on CIFAR-10:**
   ```bash
   python run_all.py --datasets cifar10 --use_attention
   ```
   This will train both schedules for 150 epochs with attention, sample, and
   compute FID/IS.  Expected runtime: several hours with a GPU.

3. **Compare results** — the key comparison is:

   | | No attention | With attention |
   |---|---|---|
   | Linear FID | 86.83 | ? |
   | Cosine FID | 191.69 | ? |

   If the hypothesis holds: cosine FID < linear FID with attention.

4. **Run FashionMNIST with attention** (optional, for completeness):
   ```bash
   python run_all.py --datasets fashionmnist --use_attention
   ```

5. **Regenerate all figures:**
   ```bash
   python analyze.py --dataset fashionmnist --skip_fid
   python analyze.py --dataset cifar10 --skip_fid
   ```

6. **Write the report** — key narrative:
   - Section 1: DDPM background, forward/reverse process, noise schedules
   - Section 2: SNR geometry — cosine has more uniform SNR, linear front-loads noise
   - Section 3: FashionMNIST results — linear wins (too simple for cosine)
   - Section 4: CIFAR-10 without attention — cosine collapses (missing global structure)
   - Section 5: CIFAR-10 with attention — cosine recovers (self-attention enables global structure at high noise)
   - Conclusion: Architecture and dataset complexity jointly determine which schedule wins

---

## File locations

```
/home/liamf/Gen-AI-Final-Project/
  ├── data.py, schedule.py, diffusion.py, model.py
  ├── train.py, sample.py, analyze.py
  ├── experiment.py, run_all.py
  ├── requirements.txt
  ├── README.md, CONTEXT.md
  ├── .venv/               (virtualenv, not in git)
  ├── data/                (downloaded datasets, not in git)
  ├── experiments/         (checkpoints + logs, not in git)
  │   ├── fashionmnist/linear/run_01/
  │   ├── fashionmnist/cosine/run_01/
  │   ├── cifar10/linear/run_01/      ← 150 epoch, no attention
  │   └── cifar10/cosine/run_01/      ← 150 epoch, no attention
  └── outputs/
      ├── fashionmnist/    ← figures from Phase 1
      └── cifar10/         ← figures from Phase 2 (no attention)
```

---

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
