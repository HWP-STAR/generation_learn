import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# 1. 超参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 20          # 潜在向量的维度，相当于“压缩后的特征数量”
batch_size = 128
epochs = 30
learning_rate = 1e-3

# 2. 数据预处理：将 MNIST 图像转为 28x28 的像素张量，值归一化到 [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),           # 将 PIL 图像或 numpy 数组转为 [0,1] 的张量
])

# 下载 MNIST 训练集和测试集
train_dataset = datasets.MNIST(root='../../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../../data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('='*30)
print('data load')
#3 VAE model
class VAE(nn.Module):
    def __init__(self,latent_dim=20):
        super().__init__()
        self.latent_dim=latent_dim
    
        # ----- 编码器 (Encoder) -----
        # 输入：28x28 的图像 -> 展平为 784 维向量
        self.fc1=nn.Linear(784,400)
        # mu and log var**2
        self.fc_mean=nn.Linear(400,latent_dim)
        self.fc_logvar=nn.Linear(400,latent_dim)

        #decoder
        self.fc3=nn.Linear(latent_dim,400)
        self.fc4=nn.Linear(400,784) # 输出重构的 784 维图像（像素值经 sigmoid 压缩到 [0,1]）

    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std) #noisy
        z=mu+eps*std
        return z

    def forward(self,x):
        # 将图像展平为 784 维向量
        x = x.view(-1, 784)
        #encode
        h=torch.relu(self.fc1(x))
        mu=self.fc_mean(h)
        logvar=self.fc_logvar(h)

        # ----- 采样（重参数化）-----
        z=self.reparameterize(mu,logvar)
        #decode
        h_dec=torch.relu(self.fc3(z))
        recon_x=torch.sigmoid(self.fc4(h_dec))
        return recon_x,mu,logvar

# 4. VAE 的损失函数 = 重构损失 + KL 散度

def vae_loss(recon_x,x,mu,logvar):
    """
    :param recon_x: 解码器重构的图像 (batch, 784)
    :param x: 原始图像 (batch, 784)
    :param mu: 编码器输出的均值 (batch, latent_dim)
    :param logvar: 编码器输出的对数方差 (batch, latent_dim)
    :return: 总损失 (标量)
    """
    # 重构损失：使用二值交叉熵（因为图像像素为[0,1]的二值概率）
    # 等价于负对数似然： -E_{z~q(z|x)}[log p(x|z)]
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL 散度：让 q(z|x) 尽可能接近标准正态分布 N(0,I)
    # 解析公式：KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 详细推导参见原论文 Appendix B
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE+KLD
#train function
def train(model,dataloader,optimizer,epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        # 前向传播：得到重构图像、均值、对数方差
        recon_batch, mu, logvar = model(data)
        # 计算损失
        loss = vae_loss(recon_batch, data.view(-1, 784), mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 每 100 个 batch 打印一次进度
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.4f}')

    avg_loss = train_loss / len(dataloader.dataset)
    print(f'====> Epoch {epoch} Average loss: {avg_loss:.4f}')

# 6. 测试函数（计算测试集上的平均损失，并保存生成图像）
def test(model, dataloader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():          # 测试阶段不计算梯度，节省内存和计算
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss(recon_batch, data.view(-1, 784), mu, logvar).item()
            
            # 只保存第一张 batch 的重构图像作为可视化样例
            if i == 0:
                n = min(data.size(0), 8)
                # 将原始图像和重构图像拼接在一起，便于比较
                comparison = torch.cat([data[:n], recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           f'results/reconstruction_{epoch}.png',
                           nrow=n)

    test_loss /= len(dataloader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

# 7. 生成新图像：从标准正态分布采样潜在向量，通过解码器生成图像
def generate(model,epoch,num_samples=64):
    model.eval()
    z=torch.randn(num_samples,latent_dim).to(device)

    with torch.no_grad():
        generated=model.decode(z)

    # 将生成的向量 reshape 为图像格式 (batch, 1, 28, 28)
    generated = generated.view(num_samples, 1, 28, 28)
    save_image(generated.cpu(), f'results/generated_{epoch}.png', nrow=8)

def decode(self,z):
    h=torch.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h))

VAE.decode=decode

if __name__=="__main__":
    # 创建 results 文件夹用于保存中间结果
    os.makedirs("results", exist_ok=True)
    model=VAE(latent_dim=latent_dim).to(device)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader, epoch)
        # 每 5 个 epoch 生成一次随机图像
        if epoch % 5 == 0:
            generate(model, epoch)

    # 训练结束后额外生成一批最终图像
    generate(model, epochs)
    print("训练完成！生成的图像保存在 results 文件夹中。")
