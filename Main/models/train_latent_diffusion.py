import argparse
import torch
from torch.utils.data import DataLoader
from dataset import ChannelDataset
from models.weighted_vae import WeightedVAE
from models.latent_diffusion import LatentDiffusionModel, train_latent_diffusion

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChannelDataset(args.hdf5_file, args.channel_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    encoder = WeightedVAE(latent_dim=args.latent_dim).to(device)
    encoder.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    encoder.eval()

    diffusion_model = LatentDiffusionModel(latent_dim=args.latent_dim).to(device)
    diffusion_model = train_latent_diffusion(
        diffusion_model, encoder, dataloader,
        args.epochs, device, timesteps=args.timesteps
    )

    torch.save(diffusion_model.state_dict(), args.output_ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_file", type=str, required=True)
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--output_ckpt", type=str, default="ldm.pth")
    parser.add_argument("--channel_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()
    main(args)
