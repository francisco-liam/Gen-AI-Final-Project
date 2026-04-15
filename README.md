# Noise Schedule Geometry in Diffusion Models

A study of how different beta schedules affect DDPM training dynamics on Fashion-MNIST.

## Week 1 — Minimal Working Pipeline

This week establishes a clean, extensible training baseline:
epsilon-prediction on Fashion-MNIST with a **linear beta schedule**.

### File overview

| File | Purpose |
|---|---|
| `data.py` | Fashion-MNIST dataloaders (normalized to \[−1, 1\]) |
| `schedule.py` | `linear_beta_schedule` — returns betas as a 1-D tensor |
| `diffusion.py` | `GaussianDiffusion` — precomputed coefficients + `q_sample` |
| `model.py` | `SmallUNet` — U-Net with sinusoidal timestep embeddings |
| `train.py` | Training loop, config, checkpointing |

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

All hyper-parameters live at the top of `train.py`:

```python
BATCH_SIZE    = 128
LEARNING_RATE = 2e-4
EPOCHS        = 10
TIMESTEPS     = 200     # T — number of diffusion steps
SAVE_EVERY    = 2       # checkpoint frequency (epochs)
NUM_WORKERS   = 2       # set to 0 on Windows if multiprocessing errors occur
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

- **Cosine schedule:** add `cosine_beta_schedule()` in `schedule.py`, retrain, compare grids
- **Loss-by-timestep analysis:** log MSE per `t` bucket to identify hard timesteps
- **SNR analysis:** `snr_t = alpha_bars / (1 - alpha_bars)` — already precomputed in `GaussianDiffusion`
- **FID / IS metrics:** evaluate sample quality quantitatively
- **DDIM sampler:** fewer steps, faster generation
