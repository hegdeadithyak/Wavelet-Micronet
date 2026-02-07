import onnxruntime as ort
import numpy as np
import time
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

onnx_path = "./wavelet_model.onnx"
data_dir  = "./dataset/test"
batch_size = 1   
img_size = 128


sess = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

input_name = sess.get_inputs()[0].name

print("ONNX loaded")

tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor()
])

test_ds = datasets.ImageFolder(data_dir+"/test", tf)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print("Test samples:", len(test_ds))

correct = 0
total = 0
times = []

for i,(x,y) in enumerate(test_loader):
    if i>20: break
    inp = x.numpy()
    sess.run(None,{input_name:inp})

for x,y in test_loader:
    inp = x.numpy()

    start = time.time()
    out = sess.run(None,{input_name:inp})
    end = time.time()

    times.append(end-start)

    pred = np.argmax(out[0],axis=1)
    correct += (pred==y.numpy()).sum()
    total += y.size(0)


acc = correct/total

avg_time = np.mean(times)
fps = 1/avg_time

print("\n==============================")
print("TEST ACCURACY:", acc)
print("Avg inference time per image: %.4f sec" % avg_time)
print("FPS: %.2f" % fps)
print("==============================")


size_mb = os.path.getsize(onnx_path)/1024/1024
print("ONNX model size: %.2f MB" % size_mb)
