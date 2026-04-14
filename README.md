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

### Extension points (Week 2+)

- **Cosine schedule:** add `cosine_beta_schedule()` in `schedule.py` and swap it in `train.py`
- **Loss-by-timestep analysis:** log `loss.item()` per `t` value inside the batch loop
- **Reverse sampling / generation:** implement `p_sample` in `diffusion.py`
- **SNR analysis:** `snr_t = alpha_bars / (1 - alpha_bars)` — already precomputed in `GaussianDiffusion`
- **Gradient norm logging:** `torch.nn.utils.clip_grad_norm_` before `optimizer.step()`

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
