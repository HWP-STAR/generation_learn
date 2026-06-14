import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 漂亮的进度条
import os

# 设备配置（自动选择GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device:{device}')
# ---- DDPM 核心超参数 ----
T = 300  # 扩散总步数，越大则加噪过程越平滑，但训练耗时也会增加
latent_dim = 100  # 这是为了和VAE对齐，实际这里用不到，保留以备后用
image_size = 28   # MNIST 图片尺寸
channels = 1      # MNIST 是单通道灰度图

# ---- 噪声调度 (Beta Schedule) ----
# β_t 是一个线性增长的序列，控制每一步添加噪声的强度。
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(device)

# 预计算一些辅助变量
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # α 的累积乘积
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # 用于 q(x_t | x_0) 的系数
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)  # 用于噪声项

# 采样时用的参数（后验 q(x_{t-1} | x_t, x_0) 的系数）
posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod) / (1.0 - alphas_cumprod)
posterior_mean_coef2 = (1.0 - alphas) * torch.sqrt(alphas_cumprod) / (1.0 - alphas_cumprod)

# 数据预处理：将图片像素归一化到 [-1, 1] 区间
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 均值0.5，标准差0.5 -> 输入范围[-1, 1]
])

train_dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../../data', train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def q_sample(x_0, t, noise=None):
    """
    正向过程：根据给定的时间步 t，直接计算出加噪后的图像 x_t。
    绕过迭代的马尔可夫链，直接实现 q(x_t | x_0)。

    公式: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε

    参数:
        x_0: 原始清晰图像 (batch, channels, H, W)
        t:  时间步 (batch, )
        noise: 可选的噪声（如果传入噪声，模型预测这个噪声）

    返回:
        x_t: 加噪后的图像
        noise: 实际添加的噪声（用于训练时的损失计算）
    """
    if noise is None:
        noise = torch.randn_like(x_0)  # 从标准正态分布采样噪声

    # 根据 t 索引对应的累积系数（广播至与图像同维度）
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t, noise

class ResidualBlock(nn.Module):
    """带有残差连接的基本卷积模块，提高深层网络的训练稳定性。"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 时间步嵌入的处理层（将时间信息融入特征）
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),           # 平滑的激活函数
                nn.Linear(time_emb_dim, out_channels)
            )

        # 输入输出通道数不一致时，通过1x1卷积调整维度
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb=None):
        # 第一层卷积
        out = F.silu(self.bn1(self.conv1(x)))
        # 第二层卷积
        out = self.bn2(self.conv2(out))
        # 如果提供了时间嵌入，则将时间信息加到特征图上
        if self.time_mlp is not None and t_emb is not None:
            out = out + self.time_mlp(t_emb)[:, :, None, None]
        # 残差连接
        out = out + self.residual(x)
        return F.silu(out)


class SimpleUNet(nn.Module):
    """
    简易的 UNet 模型（适用于 28x28 的 MNIST）。
    用于预测每一步添加的噪声。
    """
    def __init__(self, time_emb_dim=128):
        super().__init__()
        # 时间步编码器（将时间步 t 转换为特征向量）
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 编码器（下采样）
        self.enc1 = ResidualBlock(1, 32, time_emb_dim)
        self.enc2 = ResidualBlock(32, 64, time_emb_dim)
        self.pool = nn.MaxPool2d(2)   # 下采样，尺寸缩小一半

        # 解码器（上采样）
        self.dec1 = ResidualBlock(64, 64, time_emb_dim)
        self.dec2 = ResidualBlock(64, 32, time_emb_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()   # 输出范围为 [-1, 1]，与输入数据范围一致
        )

    def forward(self, x, t):
        """
        参数:
            x: 带噪图像 (batch, 1, 28, 28)
            t: 时间步 (batch, ) 标量值

        返回:
            预测的噪声 (batch, 1, 28, 28)
        """
        # 编码时间步（归一化到 [0, 1] 区间，以便于网络学习）
        t_emb = self.time_mlp(t.view(-1, 1) / 100.0)

        # 编码路径
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)

        # 解码路径
        d1 = self.dec1(e2, t_emb)
        d2 = self.dec2(self.upsample(d1), t_emb)

        # 输出
        out = self.out(d2)
        return out


@torch.no_grad()
def p_sample(model, x_t, t):
    """
    执行单步逆向去噪：从 x_t 预测 x_{t-1}。

    公式: x_{t-1} = 1/√(α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ(x_t, t))

    参数:
        model: 训练好的噪声预测模型
        x_t: 当前步骤的带噪图像
        t:  当前时间步

    返回:
        比 x_t 更清晰的图像 x_{t-1}
    """
    # 预测当前步的噪声
    predicted_noise = model(x_t, t)

    # 取当前时间步对应的系数（单标量）
    beta_t = betas[t]
    alpha_t = alphas[t]
    sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)  # 1/√α_t
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]

    # 根据 DDPM 论文公式计算均值
    mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)

    # 对于最后一步 t=0，不需要添加噪声
    if t == 0:
        return mean
    else:
        # 采样噪声，方差为 β_t
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * noise


@torch.no_grad()
def sample(model, n_samples=64):
    """
    从纯噪声开始，逐步生成 n_samples 张新图片。

    参数:
        model: 训练好的去噪模型
        n_samples: 要生成的图片数量

    返回:
        张量 (n_samples, 1, 28, 28)，值域 [-1, 1]
    """
    # 1. 从标准正态分布采样初始噪声 x_T
    x_t = torch.randn(n_samples, 1, image_size, image_size).to(device)

    # 2. 逐步进行逆向去噪
    for t in tqdm(reversed(range(T)), desc="Sampling"):
        # 创建当前步的张量 (batch, )，所有样本共享同一个 t
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t_tensor)

    return x_t


def save_images(images, path, nrow=8):
    """
    将生成的多张图片拼接成网格并保存。
    """
    # 将值从 [-1, 1] 反归一化到 [0, 1]，以便保存
    images = (images + 1.0) / 2.0
    grid = utils.make_grid(images, nrow=nrow)
    utils.save_image(grid, path)


# 初始化模型、优化器
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# 训练参数
epochs = 50
os.makedirs("ddpm_results", exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)  # (batch, 1, 28, 28)

        # 随机采样时间步 t 和噪声 ε
        batch_size = images.shape[0]
        t = torch.randint(0, T, (batch_size,), device=device).long()
        noise = torch.randn_like(images)

        # 正向扩散：生成带噪图像 x_t
        x_t, _ = q_sample(images, t, noise)

        # 模型预测噪声
        predicted_noise = model(x_t, t)

        # 计算损失 (MSE)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

    # 每 10 个 epoch 保存一次生成的图片
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            sampled_imgs = sample(model, n_samples=64)
            save_images(sampled_imgs, f"ddpm_results/sample_epoch_{epoch+1}.png")
            print(f"Saved sample at epoch {epoch+1}")


# 训练结束后，从零生成一批全新的手写数字图片
model.eval()
final_samples = sample(model, n_samples=64)
save_images(final_samples, "ddpm_results/final_generation.png")
print("Final images saved!")

# 展示结果（如果你在 Jupyter Notebook 中运行）
def show_image_grid(images, nrow=8):
    images = (images + 1.0) / 2.0
    grid = utils.make_grid(images, nrow=nrow)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.show()

show_image_grid(final_samples)


