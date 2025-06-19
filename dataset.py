# models/train_vae.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import ChannelDataset, calculate_dataset_density
from models.weighted_vae import WeightedVAE, train_vae

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ch in [0, 1, 2]:
        channel_name = ["ECAL", "HCAL", "Tracks"][ch]
        print(f"\n--- Training VAE for Channel: {channel_name} ---")

        dataset = ChannelDataset(args.hdf5_file, ch)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        density = calculate_dataset_density(dataloader, threshold=args.threshold, device=device)

        model = WeightedVAE(latent_dim=args.latent_dim).to(device)
        model = train_vae(model, dataloader, args.epochs, device,
                          args.lr, args.step_size, args.lr_gamma,
                          threshold=args.threshold, dataset_density=density,
                          weight_constant=args.weight_constant, kl_weight=args.kl_weight)

        torch.save(model.state_dict(), f"{args.save_dir}/vae_{channel_name}.pth")
        print(f"Model saved to {args.save_dir}/vae_{channel_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='saved_models')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--kl_weight', type=float, default=1e-3)
    parser.add_argument('--weight_constant', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
