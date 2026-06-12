import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from models_cifar import CNNVAE, vae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
batch_size = 128

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR100(root='../../data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR100(root='../../data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ========= Phase 1: 无监督训练 CNN VAE 50 epochs =========
vae = CNNVAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

print("=" * 60)
print("Exp1-CIFAR Phase 1: Training CNN VAE for 50 epochs")
print("=" * 60)

vae_losses = []
for epoch in range(1, 51):
    vae.train()
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    vae_losses.append(avg_loss)
    print(f"Epoch {epoch:2d}/50 | VAE Loss: {avg_loss:.4f}")

torch.save(vae.state_dict(), "checkpoints/vae_cifar_50epochs.pth")
print("VAE saved to checkpoints/vae_cifar_50epochs.pth\n")

with open("results/exp1_cifar_vae_loss.txt", "w") as f:
    f.write("epoch,vae_loss\n")
    for i, loss in enumerate(vae_losses):
        f.write(f"{i+1},{loss:.4f}\n")

# ========= Phase 2: 冻结 Encoder，在z后加分类头，只训练分类头 50 epochs =========
class VAEWithClassifier(nn.Module):
    def __init__(self, vae, latent_dim, num_classes=100):
        super().__init__()
        self.vae = vae
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        return self.classifier(z)


model = VAEWithClassifier(vae, latent_dim).to(device)

for param in model.vae.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("=" * 60)
print("Exp1-CIFAR Phase 2: Training classifier (encoder frozen) for 50 epochs")
print("=" * 60)

train_losses, train_accs, test_accs = [], [], []

for epoch in range(1, 51):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = total_loss / total
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100. * correct / total
    test_accs.append(test_acc)

    print(f"Epoch {epoch:2d}/50 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

with open("results/exp1_cifar_results.txt", "w") as f:
    f.write("epoch,train_loss,train_acc,test_acc\n")
    for i in range(50):
        f.write(f"{i+1},{train_losses[i]:.4f},{train_accs[i]:.2f},{test_accs[i]:.2f}\n")
print("Results saved to results/exp1_cifar_results.txt")
