# Full training script: model, QAT prep, train loop with logging + CSV save + plotting helper
import os
import csv
import math
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_wavelets import DWTForward

class WaveletDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')

    def forward(self, x):
        yl, yh = self.dwt(x)
        yh0 = yh[0].reshape(yh[0].shape[0], yh[0].shape[1] * yh[0].shape[2], yh[0].shape[3], yh[0].shape[4])
        return torch.cat([yl, yh0], dim=1)


class DWBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act1 = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

# Squeeze-Excite (tiny)
class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Linear(c, c//r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(c//r, c)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.shape
        y = x.mean((2,3))            # [B, C]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y).view(b,c,1,1)
        return x * y

class WaveletMicroNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.wave1 = WaveletDownsample()
        self.block1 = DWBlock(32*4)
        self.se1 = SE(32*4)

        self.wave2 = WaveletDownsample()
        self.block2 = DWBlock(32*16)
        self.se2 = SE(32*16)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32*16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.wave1(x)
        x = self.block1(x)
        x = self.se1(x)
        x = self.wave2(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def set_qat_qconfig(model):
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    return model

def prepare_model_for_qat(model):
    model.train()
    set_qat_qconfig(model)
    try:
        torch.quantization.fuse_modules(model.stem, ['0','1','2'], inplace=True)
    except Exception:
        pass
    torch.quantization.prepare_qat(model, inplace=True)
    return model


def make_loaders(root: str, img_size=128, batch_size=64, num_workers=2):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "valid") if os.path.exists(os.path.join(root, "valid")) else os.path.join(root, "test")

    tf_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(6),
        transforms.ToTensor()
    ])
    tf_val = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=tf_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tf_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(train_ds.classes)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss/total, total_correct/total

def train_qat(data_root: str,
              out_dir: str = "./out",
              epochs: int = 30,
              batch_size: int = 64,
              lr: float = 1e-3,
              img_size: int = 128,
              device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    train_loader, val_loader, num_classes = make_loaders(data_root, img_size=img_size, batch_size=batch_size)
    model = WaveletMicroNet(num_classes=num_classes).to(device)
    model = prepare_model_for_qat(model)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        t0 = time.time()

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  time={elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc
            }
            torch.save(best_state, os.path.join(out_dir, "best_qat_state.pth"))

    print("Converting to quantized module (CPU) ...")
    model.cpu()
    try:
        quantized = torch.quantization.convert(model.eval(), inplace=False)
    except Exception as e:
        print("Warning: convert() raised:", e)
        quantized = None

    if quantized is not None:
        qpath = os.path.join(out_dir, "model_qat_quantized.pth")
        torch.save(quantized.state_dict(), qpath)
        total_params = sum(p.numel() for p in quantized.parameters())
        print("Quantized param count:", total_params)
        print("Estimated INT8 size (bytes):", total_params)
        print("Estimated MB (INT8):", total_params / (1024*1024))
    else:
        sdpath = os.path.join(out_dir, "best_float_state.pth")
        torch.save(best_state, sdpath)
        print("Saved best float checkpoint at", sdpath)

    csv_path = os.path.join(out_dir, "history.csv")
    keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print("Saved training history to", csv_path)
    print("Best val acc:", best_val_acc)
    return out_dir
data_dir = "./dataset"

out = train_qat(data_dir, out_dir="./wavelet_qat_out", epochs=25, batch_size=64, lr=1e-3)
