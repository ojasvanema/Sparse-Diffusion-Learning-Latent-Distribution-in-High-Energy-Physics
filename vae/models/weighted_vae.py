import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

class WeightedVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(WeightedVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(2),  # 125 -> 62

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(2),  # 62 -> 31

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(2)   # 31 -> 15
        )

        self.encoder_output_dim = 128 * 15 * 15
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.encoder_output_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 15 -> 30
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 30 -> 60
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 60 -> 120
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z).view(-1, 128, 15, 15)
        x = self.decoder(x)
        return F.pad(x, (2, 3, 2, 3))  # Pad to 125x125

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def weighted_vae_loss(recon, x, mu, logvar, threshold, dataset_density,
                      weight_constant=0.2, kl_weight=0.0001):
    if dataset_density == 0:
        high_weight = 1.0
    else:
        high_weight = (1.0 / dataset_density) * weight_constant

    low_weight = 1.0
    weight = torch.where(x > threshold,
                         torch.tensor(high_weight, device=x.device),
                         torch.tensor(low_weight, device=x.device))

    recon_loss = (weight * (recon - x) ** 2).sum() / weight.sum()
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss

def train_vae(model, dataloader, num_epochs, device, initial_lr, step_size, lr_gamma,
              threshold, dataset_density, weight_constant, kl_weight):
    optimizer = Adam(model.parameters(), lr=initial_lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_gamma)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = weighted_vae_loss(recon, batch, mu, logvar, threshold, dataset_density,
                                     weight_constant=weight_constant, kl_weight=kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        scheduler.step()

    return model

def generate_samples(model, latent_dim, num_samples, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated = model.decode(z)
    return generated.cpu().numpy()
