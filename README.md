# Noise Schedule Geometry in Diffusion Models

A study of how linear vs cosine beta schedules affect DDPM training dynamics
and sample quality on FashionMNIST and CIFAR-10.  Models are trained under
both schedules with identical settings, with optional bottleneck self-attention,
then figures and tables comparing SNR geometry, training loss, gradient
behaviour, and generated sample quality are produced automatically.

See [CONTEXT.md](CONTEXT.md) for full project history, experimental results,
and the scientific narrative.

---

## Setup on a new machine

### Prerequisites

- Python 3.10+ (project was developed on **Python 3.10.12**)
- Git
- CUDA-capable GPU strongly recommended (CPU training is very slow for CIFAR-10)

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd Gen-AI-Final-Project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `torch==2.11.0` and `torchvision==0.26.0` in `requirements.txt`
> are CUDA 12.x builds. If your new machine has a different CUDA version or
> you need CPU-only, install PyTorch separately first:
>
> ```bash
> # CPU only
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
>
> # CUDA 11.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
>
> # Then install the rest
> pip install torchmetrics[image] torch_fidelity matplotlib numpy pandas pillow scipy tqdm
> ```

### 4. Verify the install

```bash
python -c "import torch, torchvision, torchmetrics; print('torch', torch.__version__); print('GPU:', torch.cuda.is_available())"
```

### 5. Data

FashionMNIST and CIFAR-10 are downloaded automatically on first run.
They land in `data/` (which is `.gitignore`d, so you start fresh on a new
machine — that's fine, PyTorch handles it).

---

## Quick start

```bash
# Full pipeline — both datasets, both schedules, FID/IS at the end
python run_all.py

# With bottleneck self-attention (recommended for cosine to show its advantage)
python run_all.py --use_attention

# One dataset only
python run_all.py --datasets cifar10 --use_attention

# Skip training if checkpoints already exist
python run_all.py --skip_train

# Quick smoke-test (2 epochs, no FID)
python run_all.py --epochs 2 --skip_fid
```

All outputs land in `outputs/<dataset>/`.

---

## Project structure

| File | Purpose |
|---|---|
| `data.py` | Dataloaders for FashionMNIST and CIFAR-10, normalised to [−1, 1] |
| `schedule.py` | `linear_beta_schedule`, `cosine_beta_schedule`, `get_beta_schedule` |
| `diffusion.py` | `GaussianDiffusion` — forward noising + DDPM reverse denoising |
| `model.py` | `SmallUNet` with sinusoidal timestep embeddings and optional bottleneck self-attention |
| `train.py` | Training loop with CLI, experiment folders, and CSV logging |
| `sample.py` | Image generation from any saved checkpoint |
| `experiment.py` | Runs both schedules end-to-end (alternative to `run_all.py`) |
| `analyze.py` | Loads saved outputs and produces all figures and tables |
| `run_all.py` | **Single entry point** — train → sample → analyze |

---

## Running individual steps

### Training

```bash
# Train one schedule on FashionMNIST (no attention)
python train.py --schedule linear --dataset fashionmnist --run_name run_01

# Train on CIFAR-10 with bottleneck self-attention, larger model
python train.py --schedule cosine --dataset cifar10 --base_channels 64 --epochs 150 --use_attention

# Or run both schedules for a dataset in one command
python experiment.py --datasets cifar10 --use_attention
```

### Sampling

```bash
# Architecture is read from the config saved inside the .pt file
python sample.py --ckpt experiments/fashionmnist/linear/run_01/checkpoints/ckpt_epoch0100.pt
python sample.py --ckpt experiments/cifar10/cosine/run_01/checkpoints/ckpt_epoch0150.pt
```

### Analysis only (training already done)

```bash
python analyze.py --dataset fashionmnist          # with FID/IS
python analyze.py --dataset cifar10 --skip_fid    # skip FID/IS, everything else runs
python analyze.py --dataset fashionmnist --fid_samples 2000
```

### Pipeline shortcuts

```bash
# Already trained — skip retraining
python run_all.py --skip_train

# Already have samples too — analysis only
python run_all.py --skip_train --skip_sample

# One dataset, attention on
python run_all.py --datasets cifar10 --use_attention

# Quick smoke-test with 2 epochs, no FID
python run_all.py --epochs 2 --skip_fid
```

---

## Output folder structure

```
experiments/
  fashionmnist/
    linear/run_01/
      config.json
      checkpoints/ckpt_epoch0002.pt … ckpt_epoch0100.pt
      logs/loss.csv  gradnorm.csv  loss_by_t.csv
      samples/samples_latest.png  trajectory_epoch0100.png
    cosine/run_01/   (identical layout)
  cifar10/
    linear/run_01/   (identical layout)
    cosine/run_01/

outputs/
  fashionmnist/
    snr_comparison.png
    loss_comparison.png
    loss_by_t_comparison.png
    gradnorm_comparison.png
    sample_comparison.png
    fid_is_comparison.png
    summary_metrics.csv
  cifar10/
    (same files)
```

---

## Experiment configuration

Dataset-specific defaults (overridable via CLI):

| Setting | FashionMNIST | CIFAR-10 |
|---|---|---|
| Image size | 28×28, 1ch | 32×32, 3ch |
| `base_channels` | 32 | 64 |
| Epochs | 100 | 150 |
| Timesteps T | 1000 | 1000 |
| Optimizer | Adam, lr=2e-4 | Adam, lr=2e-4 |
| Batch size | 128 | 128 |
| Random seed | 42 | 42 |
| Self-attention | off by default | off by default |

All `train.py` CLI options:

```
  --schedule        linear | cosine        (default: linear)
  --dataset         fashionmnist | cifar10 (default: fashionmnist)
  --run_name        str                    (default: run_01)
  --epochs          int                    (default: 10)
  --base_channels   int                    (default: 32)
  --timesteps       int                    (default: 1000)
  --batch_size      int                    (default: 128)
  --learning_rate   float                  (default: 2e-4)
  --save_every      int                    (default: 2)
  --seed            int                    (default: 42)
  --use_attention   flag                   add bottleneck self-attention
  --experiment_root path                   (default: experiments)
```

---

## Noise schedules

### Linear (Ho et al., 2020)

$$\beta_t = \beta_{\text{start}} + \frac{t}{T-1}(\beta_{\text{end}} - \beta_{\text{start}}), \quad \beta_{\text{start}} = 10^{-4},\ \beta_{\text{end}} = 0.02$$

### Cosine (Nichol & Dhariwal, 2021)

$$f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right), \quad s = 0.008$$

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad \beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}$$

Betas are clipped to [0, 0.999] to prevent numerical instability.

---

## Forward and reverse diffusion

**Forward (noising):**

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

**Signal-to-noise ratio:**

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}$$

**Reverse (denoising) — DDPM posterior mean:**

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \cdot \hat\varepsilon_\theta(x_t, t) \right)$$

$$x_{t-1} = \mu_\theta + \sqrt{\beta_t}\, z, \quad z \sim \mathcal{N}(0, I) \quad (t > 0)$$

---

## Checkpoint format

Each `.pt` file contains:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "loss": float,       # average MSE for that epoch
    "config": { ... }    # full hyperparameters
}
```

Load with:

```python
ckpt = torch.load("checkpoints/ckpt_epoch0010.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
```

---

## Analysis outputs

`analyze.py` reads saved logs and checkpoints and writes all figures to
`outputs/<dataset>/`.  Files can be dropped directly into a report or slides.

| Figure | What it shows |
|---|---|
| `snr_comparison.png` | How quickly each schedule destroys signal — cosine preserves SNR more uniformly |
| `loss_comparison.png` | Training convergence speed and final loss for each schedule |
| `loss_by_t_comparison.png` | Which timestep range each schedule finds hardest |
| `gradnorm_comparison.png` | Gradient stability across training |
| `sample_comparison.png` | Side-by-side generated sample grids |
| `fid_is_comparison.png` | FID (lower = better) and IS (higher = better) bar charts |
| `summary_metrics.csv` | All key numbers in one table |

---

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
