import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Config:
    T: int = 300
    image_size: int = 32
    channels: int = 3
    batch_size: int = 128
    epochs: int = 15
    lr: float = 2e-4
    time_emb_dim: int = 128
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = 'linear'  # 'linear' or 'cosine'
    ema_decay: float = 0.9999
    save_every: int = 10
    sample_every: int = 10
    n_samples: int = 64
    data_root: str = '../../data'
    dataset: str = 'cifar10'       # 'mnist' or 'cifar10'
    num_workers: int = 2
    results_dir: str = 'ddpm_results'
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')


def get_beta_schedule(cfg: Config) -> torch.Tensor:
    if cfg.beta_schedule == 'linear':
        return torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
    elif cfg.beta_schedule == 'cosine':
        steps = cfg.T + 1
        s = 0.008
        t = torch.linspace(0, cfg.T, steps)
        f = torch.cos((t / cfg.T + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(max=0.999)
    else:
        raise ValueError(f'Unknown beta_schedule: {cfg.beta_schedule}')


def get_dataset(cfg: Config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * cfg.channels, (0.5,) * cfg.channels),
    ])
    if cfg.dataset == 'mnist':
        train = datasets.MNIST(root=cfg.data_root, train=True, download=True, transform=transform)
    elif cfg.dataset == 'cifar10':
        train = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=transform)
    else:
        raise ValueError(f'Unknown dataset: {cfg.dataset}')
    return train


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * torch.log(torch.tensor(10000.0)) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        ) if time_emb_dim else None

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.time_mlp is not None and t_emb is not None:
            out = out + self.time_mlp(t_emb)[:, :, None, None]
        return F.silu(out + self.shortcut(x))


class SimpleUNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.time_emb_dim
        self.time_mlp = TimeMLP(d)

        self.enc1 = ResidualBlock(cfg.channels, 64, d)
        self.enc2 = ResidualBlock(64, 128, d)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(128, 128, d)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ResidualBlock(128 + 128, 64, d)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ResidualBlock(64 + 64, 64, d)

        self.out = nn.Sequential(
            nn.Conv2d(64, cfg.channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        e1 = self.enc1(x, t_emb)
        p1 = self.pool(e1)
        e2 = self.enc2(p1, t_emb)
        p2 = self.pool(e2)

        b = self.bottleneck(p2, t_emb)

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.out(d1)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
        self.model = model

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v * (1 - self.decay)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state):
        self.decay = state['decay']
        self.shadow = state['shadow']


class DDPM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        betas = get_beta_schedule(cfg).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_0 + sqrt_one_ac * noise, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor):
        pred = model(x_t, t)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
        sqrt_one_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        mean = sqrt_recip_alpha * (x_t - (beta_t / sqrt_one_ac) * pred)

        if t[0] == 0:
            return mean
        return mean + torch.sqrt(beta_t) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model: nn.Module, n_samples: int = 64):
        x_t = torch.randn(n_samples, self.cfg.channels, self.cfg.image_size, self.cfg.image_size, device=self.device)

        for t in tqdm(reversed(range(self.cfg.T)), desc='sampling'):
            t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_tensor)
        return x_t


def save_images(images: torch.Tensor, path: str, nrow: int = 8):
    images = (images + 1.0) / 2.0
    utils.save_image(images, path, nrow=nrow)


def train_one_epoch(model, dataloader, optimizer, ddpm: DDPM, cfg: Config, epoch: int):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{cfg.epochs}')

    for images, _ in pbar:
        images = images.to(cfg.device)
        batch_size = images.shape[0]
        t = torch.randint(0, cfg.T, (batch_size,), device=cfg.device, dtype=torch.long)
        noise = torch.randn_like(images)

        x_t, _ = ddpm.q_sample(images, t, noise)
        pred = model(x_t, t)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def main():
    cfg = Config()
    print(f'Using device: {cfg.device}')
    os.makedirs(cfg.results_dir, exist_ok=True)

    ddpm = DDPM(cfg)

    dataset = get_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model = SimpleUNet(cfg).to(cfg.device)
    ema = EMA(model, decay=cfg.ema_decay)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    start_epoch = 0
    ckpt_path = os.path.join(cfg.results_dir, 'latest.pt')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    for epoch in range(start_epoch, cfg.epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, ddpm, cfg, epoch)
        print(f'Epoch {epoch+1} Avg Loss: {avg_loss:.6f}')
        ema.update()

        torch.save({
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, ckpt_path)

        if (epoch + 1) % cfg.sample_every == 0:
            model.eval()
            ema.apply_to(model)
            sampled = ddpm.sample(model, n_samples=cfg.n_samples)
            save_images(sampled, f'{cfg.results_dir}/sample_epoch_{epoch+1}.png')
            print(f'Saved sample at epoch {epoch+1}')

    model.eval()
    ema.apply_to(model)
    final = ddpm.sample(model, n_samples=cfg.n_samples)
    save_images(final, f'{cfg.results_dir}/final_generation.png')
    torch.save(model.state_dict(), f'{cfg.results_dir}/model_final.pt')
    print('Done! Final images saved.')


if __name__ == '__main__':
    main()
