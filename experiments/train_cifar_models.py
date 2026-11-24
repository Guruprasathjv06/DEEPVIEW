"""Train high-accuracy CIFAR-10 models (Improved CNN and Basic CNN) for DEEPVISION Real-Time Prediction."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Multi-threaded CPU execution
torch.set_num_threads(os.cpu_count() or 4)

from models import get_model
from datasets.cifar10 import CIFAR10_CLASSES

def train_cifar_model(model_name="improved", epochs=4, sample_limit=20000, batch_size=64, lr=0.001):
    print(f"\n==========================================")
    print(f"Training CIFAR-10 {model_name.upper()} CNN ({epochs} epochs, {sample_limit} samples)")
    print(f"==========================================")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root="data", train=True, download=False, transform=train_transform)
    val_set = datasets.CIFAR10(root="data", train=False, download=False, transform=val_transform)

    if sample_limit and sample_limit < len(train_set):
        indices = torch.randperm(len(train_set))[:sample_limit]
        train_set = Subset(train_set, indices)

    val_indices = torch.randperm(len(val_set))[:3000]
    val_subset = Subset(val_set, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = get_model(model_name, in_channels=3, num_classes=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"cifar10_{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for b_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100

        # Fast Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for x, y in val_loader:
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                _, pred = torch.max(out, 1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "dataset": "CIFAR-10",
                "model_name": model_name,
            }, ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path} (Val Acc: {val_acc:.2f}%)")

    print(f"Done! Final Best Val Acc for CIFAR-10 {model_name}: {best_val_acc:.2f}%\n")
    return ckpt_path

if __name__ == "__main__":
    train_cifar_model(model_name="improved", epochs=4, sample_limit=25000, batch_size=64, lr=0.001)
    train_cifar_model(model_name="basic", epochs=3, sample_limit=15000, batch_size=64, lr=0.001)
