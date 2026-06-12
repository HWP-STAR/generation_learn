import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from models import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='../../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../../data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

os.makedirs("results", exist_ok=True)

model = Classifier(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("=" * 60)
print("Exp3: Standalone classifier (same structure as VAE encoder) for 50 epochs")
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

with open("results/exp3_results.txt", "w") as f:
    f.write("epoch,train_loss,train_acc,test_acc\n")
    for i in range(50):
        f.write(f"{i+1},{train_losses[i]:.4f},{train_accs[i]:.2f},{test_accs[i]:.2f}\n")
print("Results saved to results/exp3_results.txt")
