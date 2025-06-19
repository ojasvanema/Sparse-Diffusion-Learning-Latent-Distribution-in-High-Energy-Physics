import argparse
import torch
import matplotlib.pyplot as plt
from models.latent_diffusion import LatentDiffusionModel, sample_latent_diffusion
from models.weighted_vae import WeightedVAE

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = WeightedVAE(latent_dim=args.latent_dim).to(device)
    decoder.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    decoder.eval()

    diffusion_model = LatentDiffusionModel(latent_dim=args.latent_dim).to(device)
    diffusion_model.load_state_dict(torch.load(args.ldm_ckpt, map_location=device))

    samples = []
    for _ in range(args.num_samples):
        sample = sample_latent_diffusion(
            diffusion_model, decoder.decode, device,
            latent_dim=args.latent_dim, timesteps=args.timesteps
        )
        samples.append(sample.squeeze().cpu().numpy())

    fig, axes = plt.subplots(1, args.num_samples, figsize=(args.num_samples * 3, 3))
    for ax, img in zip(axes, samples):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--ldm_ckpt", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    main(args)
