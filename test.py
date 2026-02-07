import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pytorch_wavelets import DWTForward

data_dir = "/content/Wafer_Dataset/First dataset making from Third_02"
model_path = "/content/wavelet_qat_out/best_qat_state.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

class WaveletDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')

    def forward(self, x):
        yl, yh = self.dwt(x)
        yh0 = yh[0].reshape(yh[0].shape[0],
                            yh[0].shape[1]*yh[0].shape[2],
                            yh[0].shape[3],
                            yh[0].shape[4])
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

class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Linear(c, c//r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(c//r, c)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.shape
        y = x.mean((2,3))
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

def prepare_qat(model):
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(model, inplace=True)
    return model

tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

test_ds = datasets.ImageFolder(data_dir+"/test", tf)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = WaveletMicroNet(len(test_ds.classes)).to(device)
model = prepare_qat(model)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

correct = 0
total = 0
loss_total = 0
criterion = nn.CrossEntropyLoss()

all_preds = []
all_labels = []

with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        out = model(x)

        loss = criterion(out,y)
        loss_total += loss.item()*x.size(0)

        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

acc = correct/total
loss = loss_total/total

print("\nTEST ACC:", acc)
print("TEST LOSS:", loss)

size_bytes = os.path.getsize(model_path)
print("\nCheckpoint size: %.2f MB" % (size_bytes/1024/1024))

param_bytes = sum(p.numel() for p in model.parameters())
print("Param count:", param_bytes)
print("INT8 size estimate: %.2f MB" % (param_bytes/1024/1024))

model.cpu()
quantized = torch.quantization.convert(model.eval(), inplace=False)

torch.save(quantized.state_dict(), "quantized_model.pth")
qsize = os.path.getsize("quantized_model.pth")
print("\nTRUE INT8 model size: %.2f MB" % (qsize/1024/1024))

cm = confusion_matrix(all_labels, all_preds)
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=test_ds.classes,
            yticklabels=test_ds.classes,
            cmap="Blues")
plt.show()
