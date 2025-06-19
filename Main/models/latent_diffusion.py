import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.weighted_vae import WeightedVAE

# Linear beta schedule
def latent_linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

latent_timesteps = 1000
betas_latent = latent_linear_beta_schedule(latent_timesteps)
alphas_latent = 1.0 - betas_latent
alpha_bars_latent = torch.cumprod(alphas_latent, dim=0)

class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim=32, time_embedding_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.model = nn.Sequential(
            nn.Linear(latent_dim + time_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, z, t):
        t = t.float().view(-1, 1)
        time_emb = self.time_mlp(t)
        x = torch.cat([z, time_emb], dim=1)
        return self.model(x)

def forward_diffusion_latent(z0, t, device):
    alpha_bar = alpha_bars_latent.to(device)[t].view(-1, 1)
    noise = torch.randn_like(z0)
    z_t = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * noise
    return z_t, noise

def train_latent_diffusion(model, encoder, dataloader, num_epochs, device, timesteps=1000):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 10, gamma=0.1)
    mse_loss = nn.MSELoss()
    model.train()
    encoder.eval()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                mu, logvar = encoder.encode(batch)
                z0 = encoder.reparameterize(mu, logvar)

            B = z0.size(0)
            t = torch.randint(0, timesteps, (B,), device=device)
            z_t, noise = forward_diffusion_latent(z0, t, device)
            pred_noise = model(z_t, t)
            loss = mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B

        scheduler.step()
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model

@torch.no_grad()
def sample_latent_diffusion(model, decoder, device, latent_dim=32, timesteps=1000):
    model.eval()
    z = torch.randn(1, latent_dim).to(device)
    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([t], device=device)
        pred_noise = model(z, t_tensor)

        alpha = alphas_latent[t].to(device)
        alpha_bar = alpha_bars_latent[t].to(device)
        beta = betas_latent[t].to(device)

        z = (1 / math.sqrt(alpha)) * (z - ((1 - alpha) / math.sqrt(1 - alpha_bar)) * pred_noise)
        if t > 0:
            z += math.sqrt(beta) * torch.randn_like(z)

    return decoder(z)
