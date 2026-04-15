# Noise Schedule Geometry in Diffusion Models

A study of how different beta schedules affect DDPM training dynamics on Fashion-MNIST.

## Week 1 — Minimal Working Pipeline

This week establishes a clean, extensible training baseline:
epsilon-prediction on Fashion-MNIST with a **linear beta schedule**.

### File overview

| File | Purpose |
|---|---|
| `data.py` | Fashion-MNIST dataloaders (normalized to \[−1, 1\]) |
| `schedule.py` | `linear_beta_schedule`, `cosine_beta_schedule`, `get_beta_schedule` |
| `diffusion.py` | `GaussianDiffusion` — forward + reverse diffusion |
| `model.py` | `SmallUNet` — U-Net with sinusoidal timestep embeddings |
| `train.py` | Training loop with CLI, experiment folders, loss logging |
| `sample.py` | Image generation from any checkpoint |
| `experiment.py` | Runs linear vs cosine comparison end-to-end |

### Quick start

```bash
# Install dependencies (PyTorch ≥ 2.0 recommended)
pip install torch torchvision

# Run training (Fashion-MNIST downloads automatically on first run)
python train.py
```

### Expected output

A healthy Week 1 run should look roughly like:

```
Training on: cuda
Training batches per epoch: 468
Model parameters: 417,985
Epoch [  1/10]  loss: 0.9214
Epoch [  2/10]  loss: 0.6831
           checkpoint saved → checkpoints/ckpt_epoch0002.pt
Epoch [  3/10]  loss: 0.6102
...
Epoch [ 10/10]  loss: 0.4873
           checkpoint saved → checkpoints/ckpt_epoch0010.pt
Training complete.
```

**Signs training is healthy:**
- Loss drops noticeably from epoch 1 → 3 (typically ~1.0 → ~0.6)
- Loss continues decreasing, slowing toward epoch 10
- No NaN or inf values appear

**If loss stays flat or explodes:** lower the learning rate or reduce `TIMESTEPS`.

### Configuration

All hyper-parameters are configurable via CLI (see Week 3 CLI reference).
Defaults:

```python
schedule      = "linear"
batch_size    = 128
learning_rate = 2e-4
epochs        = 10
timesteps     = 200     # T — number of diffusion steps
save_every    = 2       # checkpoint frequency (epochs)
seed          = 42
```

### Checkpoint format

Each `.pt` file contains:
```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "loss": float,          # average MSE for that epoch
    "config": { ... }       # hyper-parameters used
}
```

Load with:
```python
ckpt = torch.load("checkpoints/ckpt_epoch0010.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
```

---

## Week 2 — Reverse Diffusion & Image Generation

This week implements the full DDPM reverse process so the trained model can
generate Fashion-MNIST images from pure Gaussian noise.

### New / updated files

| File | Changes |
|---|---|
| `diffusion.py` | Added `p_sample`, `sample`, `sample_with_trajectory` |
| `sample.py` | New standalone generation script |

### Quick start

```bash
# Generate 16 images using the latest checkpoint (default)
python sample.py

# Use a specific checkpoint
python sample.py --ckpt checkpoints/ckpt_epoch0006.pt
```

Outputs are written to `outputs/`:

| File | Contents |
|---|---|
| `outputs/samples_epoch0010.png` | 4×4 grid of generated images |
| `outputs/samples_latest.png` | Stable alias, always the most recent run |
| `outputs/trajectory_epoch0010.png` | Single sample denoising strip (noisy → clean) |

### Configuration

All options live at the top of `sample.py`:

```python
CHECKPOINT_PATH = "checkpoints/ckpt_epoch0010.pt"
NUM_SAMPLES     = 16      # keep a perfect square for a clean grid
TIMESTEPS       = 200     # must match the value used in train.py
OUTPUT_DIR      = "outputs"
GRID_NROW       = 4
SAVE_TRAJECTORY = True
TRAJ_EVERY      = 20      # snapshot frequency for the trajectory strip
```

### Reverse diffusion equations

The DDPM posterior mean at each step:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \cdot \hat\epsilon_\theta(x_t, t) \right)$$

Sample:

$$x_{t-1} = \mu_\theta + \sqrt{\beta_t} \cdot z, \quad z \sim \mathcal{N}(0, I) \quad (t > 0)$$
$$x_0 = \mu_\theta \quad (t = 0, \text{ no noise added})$$

### Signs generation is healthy

- Grid contains recognisable Fashion-MNIST silhouettes (tops, bags, shoes, etc.)
- Trajectory strip shows smooth noise → structure progression left to right
- No all-black, all-white, or fully uniform images

### Extension points (Week 3+)

- **Loss-by-timestep analysis:** log MSE per `t` bucket to identify hard timesteps
- **SNR analysis:** `snr_t = alpha_bars / (1 - alpha_bars)` — already precomputed in `GaussianDiffusion`
- **FID / IS metrics:** evaluate sample quality quantitatively
- **DDIM sampler:** fewer steps, faster generation

---

## Week 3 — Cosine Schedule & Reproducible Comparisons

This week adds the cosine noise schedule and the infrastructure to run fair,
reproducible comparisons between linear and cosine training runs.

### New / updated files

| File | Changes |
|---|---|
| `schedule.py` | Added `cosine_beta_schedule`, `get_beta_schedule` dispatcher |
| `train.py` | Full CLI, per-run experiment folders, JSON config, CSV loss log |
| `sample.py` | Reads schedule from checkpoint config — no longer assumes linear |
| `experiment.py` | New — runs both schedules with identical settings |

### Quick start

**Run the full comparison (recommended):**
```bash
python experiment.py
```

**Run schedules individually:**
```bash
python train.py --schedule linear --run_name run_01
python train.py --schedule cosine --run_name run_01
```

**Generate samples for a completed run:**
```bash
python sample.py --ckpt experiments/linear/run_01/checkpoints/ckpt_epoch0010.pt
python sample.py --ckpt experiments/cosine/run_01/checkpoints/ckpt_epoch0010.pt
```

### Output folder structure

```
experiments/
  linear/run_01/
    config.json                        ← full hyperparameters
    checkpoints/ckpt_epoch0002.pt
    checkpoints/ckpt_epoch0010.pt
    logs/loss.csv                      ← epoch, avg_loss (for Week 4 plots)
    samples/samples_epoch0010.png
    samples/samples_latest.png
    samples/trajectory_epoch0010.png
  cosine/run_01/
    (identical layout)
```

### Fair comparison guarantee

The following are held constant across all runs — only `--schedule` changes:

| Setting | Value |
|---|---|
| Dataset | Fashion-MNIST |
| Architecture | `SmallUNet` (base\_channels=32, time\_dim=128) |
| Timesteps T | 200 |
| Optimizer | Adam, lr=2e-4 |
| Batch size | 128 |
| Epochs | 10 |
| Random seed | 42 |

### Cosine schedule

Implemented from Nichol & Dhariwal (2021). Defines ᾱ_t as a cosine curve,
then derives β_t from consecutive ratios:

$$f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right), \quad s = 0.008$$

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad \beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}$$

Betas are clipped to [0, 0.999] to prevent numerical instability.

### train.py CLI reference

```
python train.py [options]

  --schedule       linear | cosine          (default: linear)
  --run_name       name for this run        (default: run_01)
  --epochs         int                      (default: 10)
  --timesteps      int                      (default: 200)
  --batch_size     int                      (default: 128)
  --learning_rate  float                    (default: 2e-4)
  --save_every     int                      (default: 2)
  --seed           int                      (default: 42)
  --experiment_root  path                   (default: experiments)
```

### Extension points (Week 4+)

- **Loss curve comparison:** load both `logs/loss.csv` files and plot on the same axes
- **SNR analysis:** `snr_t = alpha_bars / (1 - alpha_bars)` — plot linear vs cosine SNR curves
- **Loss-by-timestep analysis:** log MSE per `t` bucket to identify where each schedule struggles
- **Gradient norm logging:** add `clip_grad_norm_` and log norms per epoch
- **FID / IS metrics:** quantitative sample quality comparison
