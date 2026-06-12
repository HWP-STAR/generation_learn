import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 3x32x32 -> 128x4x4
        self.enc_conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(128)

        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder: latent_dim -> 3x32x32
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(32)
        self.dec_conv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        h = torch.relu(self.enc_bn1(self.enc_conv1(x)))
        h = torch.relu(self.enc_bn2(self.enc_conv2(h)))
        h = torch.relu(self.enc_bn3(self.enc_conv3(h)))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 128, 4, 4)
        h = torch.relu(self.dec_bn1(self.dec_conv1(h)))
        h = torch.relu(self.dec_bn2(self.dec_conv2(h)))
        h = torch.sigmoid(self.dec_conv3(h))
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = torch.relu(self.bn3(self.conv3(h)))
        h = h.view(h.size(0), -1)
        return self.fc(h)
