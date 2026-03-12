#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


def list_available_models():
    return ["vae", "dcgan", "diffusion"]


class ConvVAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z).view(-1, 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device)
        return self.decode(z)


def vae_loss(recon, target, mu, logvar):
    recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + 0.1 * kl
    return total, {"loss": total.item(), "recon_loss": recon_loss.item(), "kl_loss": kl.item()}


class Generator(nn.Module):
    def __init__(self, latent_dim=32, image_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class TimeConditionedDenoiser(nn.Module):
    def __init__(self, image_channels=3, num_steps=20, time_dim=16, hidden_dim=64):
        super().__init__()
        self.time_embedding = nn.Embedding(num_steps, time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Conv2d(image_channels + time_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, image_channels, 3, padding=1),
        )

    def forward(self, x_t, t):
        t_emb = self.time_proj(self.time_embedding(t))
        t_map = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_t.size(2), x_t.size(3))
        return self.net(torch.cat([x_t, t_map], dim=1))


def make_diffusion_schedule(num_steps, device):
    beta = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return {"beta": beta, "alpha": alpha, "alpha_bar": alpha_bar, "num_steps": num_steps}


def q_sample(x0, t, noise, schedule):
    alpha_bar = schedule["alpha_bar"][t].view(-1, 1, 1, 1)
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise


def diffusion_loss(model, x0, schedule):
    batch_size = x0.size(0)
    t = torch.randint(0, schedule["num_steps"], (batch_size,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred_noise = model(x_t, t)
    loss = F.mse_loss(pred_noise, noise)
    return loss, {"loss": loss.item(), "noise_mse": loss.item()}


@torch.no_grad()
def sample_diffusion(model, schedule, num_samples, image_channels, image_size, device):
    x = torch.randn(num_samples, image_channels, image_size, image_size, device=device)
    beta = schedule["beta"]
    alpha = schedule["alpha"]
    alpha_bar = schedule["alpha_bar"]
    for step in reversed(range(schedule["num_steps"])):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        pred_noise = model(x, t)
        a = alpha[step]
        a_bar = alpha_bar[step]
        x = (x - (1 - a) / torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a)
        if step > 0:
            x = x + torch.sqrt(beta[step]) * torch.randn_like(x)
    return x.clamp(-1.0, 1.0)


def build_model_bundle(config, model_name, device):
    if model_name == "vae":
        model = ConvVAE(image_channels=config.image_channels, latent_dim=config.latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        return {"name": model_name, "model": model, "optimizer": optimizer}

    if model_name == "dcgan":
        generator = Generator(latent_dim=config.latent_dim, image_channels=config.image_channels).to(device)
        discriminator = Discriminator(image_channels=config.image_channels).to(device)
        criterion = nn.BCEWithLogitsLoss()
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        return {
            "name": model_name,
            "generator": generator,
            "discriminator": discriminator,
            "criterion": criterion,
            "g_optimizer": g_optimizer,
            "d_optimizer": d_optimizer,
        }

    if model_name == "diffusion":
        model = TimeConditionedDenoiser(
            image_channels=config.image_channels,
            num_steps=config.diffusion_steps,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        schedule = make_diffusion_schedule(config.diffusion_steps, device)
        return {"name": model_name, "model": model, "optimizer": optimizer, "schedule": schedule}

    raise ValueError(f"Unsupported model: {model_name}")


def train_step(bundle, real_images, config, device):
    model_name = bundle["name"]
    real_images = real_images.to(device)

    if model_name == "vae":
        model = bundle["model"]
        optimizer = bundle["optimizer"]
        recon, mu, logvar = model(real_images)
        loss, metrics = vae_loss(recon, real_images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return metrics

    if model_name == "dcgan":
        generator = bundle["generator"]
        discriminator = bundle["discriminator"]
        criterion = bundle["criterion"]
        g_optimizer = bundle["g_optimizer"]
        d_optimizer = bundle["d_optimizer"]
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
        fake_images = generator(noise)

        d_optimizer.zero_grad()
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images.detach())
        d_loss_real = criterion(real_logits, real_labels)
        d_loss_fake = criterion(fake_logits, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        fake_logits_for_g = discriminator(fake_images)
        g_loss = criterion(fake_logits_for_g, real_labels)
        g_loss.backward()
        g_optimizer.step()

        return {
            "loss": d_loss.item() + g_loss.item(),
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "real_score": torch.sigmoid(real_logits).mean().item(),
            "fake_score": torch.sigmoid(fake_logits).mean().item(),
        }

    if model_name == "diffusion":
        model = bundle["model"]
        optimizer = bundle["optimizer"]
        loss, metrics = diffusion_loss(model, real_images, bundle["schedule"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return metrics

    raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def evaluate_step(bundle, real_images, config, device):
    model_name = bundle["name"]
    real_images = real_images.to(device)

    if model_name == "vae":
        model = bundle["model"]
        recon, mu, logvar = model(real_images)
        _, metrics = vae_loss(recon, real_images, mu, logvar)
        return metrics

    if model_name == "dcgan":
        generator = bundle["generator"]
        discriminator = bundle["discriminator"]
        criterion = bundle["criterion"]
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images)
        d_loss = criterion(real_logits, real_labels) + criterion(fake_logits, fake_labels)
        g_loss = criterion(fake_logits, real_labels)
        return {
            "loss": d_loss.item() + g_loss.item(),
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "real_score": torch.sigmoid(real_logits).mean().item(),
            "fake_score": torch.sigmoid(fake_logits).mean().item(),
        }

    if model_name == "diffusion":
        model = bundle["model"]
        _, metrics = diffusion_loss(model, real_images, bundle["schedule"])
        return metrics

    raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def sample_images(bundle, config, num_samples, device):
    model_name = bundle["name"]
    if model_name == "vae":
        return bundle["model"].sample(num_samples, device)
    if model_name == "dcgan":
        noise = torch.randn(num_samples, config.latent_dim, 1, 1, device=device)
        return bundle["generator"](noise)
    if model_name == "diffusion":
        return sample_diffusion(
            bundle["model"],
            bundle["schedule"],
            num_samples,
            config.image_channels,
            config.image_size,
            device,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def save_bundle(bundle, path):
    if bundle["name"] == "dcgan":
        torch.save(
            {
                "name": bundle["name"],
                "generator": bundle["generator"].state_dict(),
                "discriminator": bundle["discriminator"].state_dict(),
                "g_optimizer": bundle["g_optimizer"].state_dict(),
                "d_optimizer": bundle["d_optimizer"].state_dict(),
            },
            path,
        )
        return

    torch.save(
        {
            "name": bundle["name"],
            "model": bundle["model"].state_dict(),
            "optimizer": bundle["optimizer"].state_dict(),
        },
        path,
    )


def load_bundle(bundle, path, device):
    state = torch.load(path, map_location=device)
    if bundle["name"] == "dcgan":
        bundle["generator"].load_state_dict(state["generator"])
        bundle["discriminator"].load_state_dict(state["discriminator"])
        return bundle

    bundle["model"].load_state_dict(state["model"])
    return bundle