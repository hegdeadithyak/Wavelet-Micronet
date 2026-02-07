import torch
import torch.nn as nn
import onnx
import os
from pytorch_wavelets import DWTForward

device="cpu"
model_path="/content/wavelet_qat_out/best_qat_state.pth"
onnx_path="/content/wavelet_model.onnx"

# -------- ORIGINAL TRAINING MODEL --------
class WaveletDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')

    def forward(self,x):
        yl,yh=self.dwt(x)
        yh0=yh[0].reshape(yh[0].shape[0],
                          yh[0].shape[1]*yh[0].shape[2],
                          yh[0].shape[3],
                          yh[0].shape[4])
        return torch.cat([yl,yh0],dim=1)

class DWBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.dw = nn.Conv2d(c,c,3,padding=1,groups=c,bias=False)
        self.bn1=nn.BatchNorm2d(c)
        self.act1=nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(c,c,1,bias=False)
        self.bn2=nn.BatchNorm2d(c)
        self.act2=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.dw(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.pw(x)
        x=self.bn2(x)
        x=self.act2(x)
        return x

class SE(nn.Module):
    def __init__(self,c,r=4):
        super().__init__()
        self.fc1=nn.Linear(c,c//r)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(c//r,c)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        b,c,h,w=x.shape
        y=x.mean((2,3))
        y=self.fc1(y)
        y=self.relu(y)
        y=self.fc2(y)
        y=self.sig(y).view(b,c,1,1)
        return x*y

class WaveletMicroNet(nn.Module):
    def __init__(self,num_classes=9):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.wave1=WaveletDownsample()
        self.block1=DWBlock(32*4)
        self.se1=SE(32*4)

        self.wave2=WaveletDownsample()
        self.block2=DWBlock(32*16)
        self.se2=SE(32*16)

        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(32*16,num_classes)

    def forward(self,x):
        x=self.stem(x)
        x=self.wave1(x)
        x=self.block1(x)
        x=self.se1(x)
        x=self.wave2(x)
        x=self.block2(x)
        x=self.se2(x)
        x=self.pool(x).flatten(1)
        return self.fc(x)

# -------- LOAD MODEL --------
model=WaveletMicroNet()

# IMPORTANT: load with strict=False
checkpoint=torch.load(model_path,map_location=device)
model.load_state_dict(checkpoint["model_state_dict"],strict=False)

model.eval()
model.cpu()

print("Weights loaded successfully")

# -------- EXPORT ONNX --------
dummy=torch.randn(1,1,128,128)

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}}
)

print("ONNX saved:",onnx_path)

# -------- VERIFY --------
onnx_model=onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX verified")

size=os.path.getsize(onnx_path)/1024/1024
print("ONNX size: %.2f MB" % size)