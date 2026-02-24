# Physics-Informed Diffusion for CFD

A clean reimplementation of diffusion-based fluid super-resolution for computational fluid dynamics (CFD). Reconstructs high-resolution flow fields from low-resolution observations using physics-informed denoising diffusion.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test on CPU (creates synthetic data)
python test_cpu.py

# 3. Train and sample (GPU)
./scripts/run_full_pipeline.sh --quick  # 2 epochs test
./scripts/run_full_pipeline.sh --epochs 100  # full training
```

## Setup

### Requirements

- Python 3.8+
- PyTorch 1.7+ (with CUDA for GPU training)
- ~2GB GPU memory for training (64x64), ~8GB for 256x256

### Installation

```bash
# Clone or copy the repo
cd diffusion_cfd_clean

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_cpu.py
```

### Data Setup

**Option 1: Use synthetic data for testing**
```bash
python test_cpu.py  # Creates data/test/synthetic_test.npy
```

**Option 2: Use real Kolmogorov flow data**

Download the data and update `configs/default.yaml`:
```yaml
data:
  data_dir: "/path/to/kf_2d_re1000_256_40seed.npy"
  image_size: 256
```

Data format: NumPy array with shape `(num_seeds, num_timesteps, H, W)`

## Usage

### Training

```bash
# Basic training
python main.py --config configs/default.yaml --mode train

# Resume from checkpoint
python main.py --config configs/default.yaml --mode train \
    --checkpoint checkpoints/checkpoint_step10000.pth

# Specify device
python main.py --config configs/default.yaml --mode train --device cuda:1
```

### Sampling / Inference

```bash
python main.py --config configs/default.yaml --mode sample \
    --checkpoint checkpoints/final_model.pth
```

### Full Pipeline (Train + Sample)

```bash
# Quick test (2 epochs)
./scripts/run_full_pipeline.sh --quick

# Full training
./scripts/run_full_pipeline.sh --epochs 100

# Custom config
./scripts/run_full_pipeline.sh --config configs/my_config.yaml --epochs 50
```

## Configuration

All settings are in `configs/default.yaml`:

```yaml
# Data
data:
  data_dir: "./data/test/synthetic_test.npy"  # Path to data
  image_size: 64          # 64 for testing, 256 for full
  channels: 3             # 3 consecutive timesteps
  train_split: 0.9

# Model
model:
  type: "conditional"     # "simple" or "conditional" (physics-guided)
  base_channels: 64       # 32 for testing, 64 for full
  channel_mult: [1, 2, 4] # Channel multipliers per level
  num_res_blocks: 2
  attention_resolutions: [16]
  dropout: 0.1
  num_groups: 8

# Diffusion
diffusion:
  num_timesteps: 1000
  beta_schedule: "linear"
  beta_start: 0.0001
  beta_end: 0.02

# Training
training:
  batch_size: 2           # Increase for GPU
  num_epochs: 100
  learning_rate: 0.0002
  grad_clip: 1.0
  use_ema: true
  save_freq: 10000
  log_freq: 100

# Sampling
sampling:
  batch_size: 4
  num_steps: 50           # DDIM steps
  eta: 0.0                # 0=DDIM (deterministic), 1=DDPM (stochastic)
  physics_weight: 0.0     # Physics guidance strength

# Physics (Kolmogorov flow)
physics:
  reynolds_number: 1000
  dt: 0.03125

# Paths
paths:
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
  output_dir: "./outputs"
```

### Config for Different Scenarios

**Quick CPU test:**
```yaml
data:
  image_size: 64
model:
  base_channels: 32
  channel_mult: [1, 2, 4]
training:
  batch_size: 1
  num_epochs: 2
```

**Full GPU training (256x256):**
```yaml
data:
  image_size: 256
model:
  base_channels: 64
  channel_mult: [1, 1, 1, 2]
training:
  batch_size: 4
  num_epochs: 100
```

## Project Structure

```
diffusion_cfd_clean/
├── configs/
│   └── default.yaml         # Configuration file
├── data/
│   ├── __init__.py
│   └── dataset.py           # FlowDataset, InferenceDataLoader
├── models/
│   ├── __init__.py
│   ├── unet.py              # UNet architecture
│   ├── diffusion_utils.py   # Beta schedules, sampling
│   └── physics.py           # Vorticity residual (FFT-based)
├── utils/
│   ├── __init__.py
│   ├── config.py            # Config loading
│   └── ema.py               # Exponential Moving Average
├── scripts/
│   ├── train.sh
│   ├── sample.sh
│   └── run_full_pipeline.sh # Full train+sample pipeline
├── main.py                  # Entry point
├── train.py                 # Training loop
├── sample.py                # Sampling/inference
├── test_cpu.py              # CPU verification test
├── requirements.txt
└── README.md
```

## Output

After training and sampling:

```
checkpoints/
├── checkpoint_step10000.pth
├── checkpoint_step20000.pth
└── final_model.pth

logs/
└── events.out.tfevents.*    # TensorBoard logs

outputs/
├── results.npz              # predictions, targets, inputs arrays
├── visualization_0.png
├── visualization_1.png
└── ...
```

### View TensorBoard Logs

```bash
tensorboard --logdir logs/
```

### Load Results

```python
import numpy as np

results = np.load("outputs/results.npz")
predictions = results["predictions"]  # (N, 3, H, W)
targets = results["targets"]          # (N, 3, H, W)
inputs = results["inputs"]            # (N, 3, H, W) low-res
```

## Model Architecture

**UNet with:**
- Sinusoidal timestep embeddings
- Residual blocks with time injection
- Self-attention at specified resolutions
- Skip connections between encoder/decoder
- Optional physics gradient conditioning

**Parameters:** ~760K (small), ~7M (full 256x256)

## Physics-Informed Conditioning

The model can use physics-based guidance during training and sampling:

**Vorticity transport equation:**
```
∂ω/∂t + u·∇ω = ν∇²ω + f
```

- Uses FFT-based spectral derivatives (accurate for periodic domains)
- Computes vorticity residual as physics loss
- Gradient of residual used for conditioning

**Enable physics conditioning:**
```yaml
model:
  type: "conditional"  # Uses physics gradient as input

sampling:
  physics_weight: 0.1  # Add physics guidance during sampling
```

## Troubleshooting

**CUDA out of memory:**
- Reduce `training.batch_size`
- Reduce `model.base_channels` (64 → 32)
- Reduce `data.image_size` (256 → 128)

**"Expected number of channels divisible by num_groups":**
- Ensure `model.base_channels` is divisible by `model.num_groups`
- Default: 64 channels, 8 groups (64/8=8 ✓)

**Training loss not decreasing:**
- Check data path is correct
- Try lower learning rate (0.0001)
- Increase `training.num_epochs`

**Sampling produces noise:**
- Train for more epochs
- Use more sampling steps (`sampling.num_steps: 100`)
- Try DDPM instead of DDIM (`sampling.eta: 1.0`)

## License

Same as original repository.

## Citation

If using this code, please cite the original paper:
```
Physics-informed diffusion model for high-fidelity flow field reconstruction
Journal of Computational Physics
```
