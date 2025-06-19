# models/sample_vae.py
import os
import argparse
import torch
import matplotlib.pyplot as plt
from models.weighted_vae import WeightedVAE, generate_samples

def plot_samples(samples, channel_name):
    import numpy as np
    fig, axs = plt.subplots(1, len(samples), figsize=(len(samples) * 2, 2))
    fig.suptitle(f"Samples - {channel_name}")
    for ax, img in zip(axs, samples):
        ax.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ch in [0, 1, 2]:
        channel_name = ["ECAL", "HCAL", "Tracks"][ch]
        print(f"\n--- Sampling for Channel: {channel_name} ---")

        model_path = os.path.join(args.model_dir, f"vae_{channel_name}.pth")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        model = WeightedVAE(latent_dim=args.latent_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        samples = generate_samples(model, args.latent_dim, args.num_samples, device)
        plot_samples(samples, channel_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='saved_models')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    main(args)
