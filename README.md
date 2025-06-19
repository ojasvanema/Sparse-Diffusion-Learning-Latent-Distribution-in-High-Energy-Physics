
# Sparse Diffusion Learning Latent Distribution in High Energy Physics

This repository implements modeling of sparse high-energy physics jet data (such as ECAL/HCAL/Tracks) using:
- **Weighted Variational Autoencoder (VAE)** with custom loss
- **Latent Diffusion Model (LDM)** on VAE's latent space

It supports modular training & inference using CLI with `argparse`.

---

## Repository Structure

```
Sparse-Diffusion-Learning-Latent-Distribution-in-High-Energy-Physics/
│
├── dataset.py                      # Dataset loader for HDF5 jet data
│
├── models/
│   ├── __init__.py
│   ├── weighted_vae.py            # Weighted VAE model & loss
│   ├── train_weighted_vae.py     # Train script for Weighted VAE (via CLI)
│   ├── sample_weighted_vae.py    # Sample/reconstruct using trained VAE
│   ├── latent_diffusion.py       # Latent Diffusion model & noise scheduling
│   ├── train_latent_diffusion.py # Train script for LDM (via CLI)
│   └── sample_latent_diffusion.py# Generate images from LDM
│
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/ojasvanema/Sparse-Diffusion-Learning-Latent-Distribution-in-High-Energy-Physics.git
cd "Sparse Diffusion Learning Latent Distribution in High Energy Physics"

# Create environment
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)

```
# Python version (informative, not enforced via pip)
# Python 3.11.11

torch==2.6.0+cu124
numpy==1.26.4
h5py==3.13.0
matplotlib==3.7.2
argparse
tqdm

# Optional for CUDA-specific install
# pip install torch==2.6.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html

```

---

## 📊 Dataset Format

We use a preprocessed HDF5 file with structure:

```python
with h5py.File('quark-gluon_data-set.hdf5', 'r') as f:
    f['X_jets']  # shape: (N, 125, 125, 3), where 3 channels = ECAL, HCAL, Tracks
```

Each channel is processed individually.

---

## Model 1: Weighted VAE

### 🏋️ Train Weighted VAE

```bash
python models/train_weighted_vae.py \
  --hdf5_file path/to/quark-gluon_data-set.hdf5 \
  --channel_idx 0 \
  --epochs 75 \
  --batch_size 256 \
  --latent_dim 32 \
  --output_ckpt vae_ECAL.pth
```

Arguments:
- `--channel_idx`: `0` = ECAL, `1` = HCAL, `2` = Tracks
- `--latent_dim`: dimension of latent space
- `--output_ckpt`: path to save model

---

###  Sample from VAE (Reconstruction)

```bash
python models/sample_weighted_vae.py \
  --hdf5_file path/to/quark-gluon_data-set.hdf5 \
  --vae_ckpt path/to/vae_ECAL.pth \
  --channel_idx 0 \
  --latent_dim 32
```

---

##  Model 2: Latent Diffusion Model (LDM)

###  Train Latent Diffusion

```bash
python models/train_latent_diffusion.py \
  --hdf5_file path/to/quark-gluon_data-set.hdf5 \
  --vae_ckpt path/to/vae_ECAL.pth \
  --channel_idx 0 \
  --output_ckpt ldm_ECAL.pth \
  --batch_size 128 \
  --epochs 30 \
  --latent_dim 32 \
  --timesteps 1000
```

Arguments:
- `--vae_ckpt`: pretrained VAE checkpoint
- `--channel_idx`: ECAL = 0, HCAL = 1, Tracks = 2
- `--timesteps`: diffusion steps (default: 1000)

---

###  Sample from LDM

```bash
python models/sample_latent_diffusion.py \
  --vae_ckpt path/to/vae_ECAL.pth \
  --ldm_ckpt path/to/ldm_ECAL.pth \
  --num_samples 5 \
  --latent_dim 32 \
  --timesteps 1000
```

---

##  Example Workflow

```bash
# Train VAE for ECAL
python models/train_weighted_vae.py --hdf5_file data.hdf5 --channel_idx 0 --output_ckpt vae_ECAL.pth

# Train Latent Diffusion on ECAL latent space
python models/train_latent_diffusion.py --hdf5_file data.hdf5 --vae_ckpt vae_ECAL.pth --channel_idx 0 --output_ckpt ldm_ECAL.pth

# Generate samples
python models/sample_latent_diffusion.py --vae_ckpt vae_ECAL.pth --ldm_ckpt ldm_ECAL.pth
```

