import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./history.csv")

def smooth(y, window=3):
    return pd.Series(y).rolling(window, min_periods=1).mean()

train_loss = smooth(df["train_loss"])
val_loss   = smooth(df["val_loss"])
train_acc  = smooth(df["train_acc"])
val_acc    = smooth(df["val_acc"])

best_epoch = df["val_loss"].idxmin()

plt.style.use("seaborn-v0_8-whitegrid")

fig = plt.figure(figsize=(12,5), dpi=120)
gs = fig.add_gridspec(1,2)

ax1 = fig.add_subplot(gs[0,0])
ax1.plot(df["epoch"], train_loss, linewidth=2.5)
ax1.plot(df["epoch"], val_loss, linewidth=2.5)

ax1.set_title("Loss Curve", fontsize=14, weight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend(["Train","Validation","Best"], frameon=True)
ax1.spines[['top','right']].set_visible(False)

ax2 = fig.add_subplot(gs[0,1])
ax2.plot(df["epoch"], train_acc, linewidth=2.5)
ax2.plot(df["epoch"], val_acc, linewidth=2.5)

ax2.set_title("Accuracy Curve", fontsize=14, weight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend(["Train","Validation","Best"], frameon=True)
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig("plot.png", dpi=300, bbox_inches="tight")
plt.show()
