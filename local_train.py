import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import random

from mia_code.config import CFG
from mia_code.train import train_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=tfm
)

n_total = CFG["n_total"]
n_target = int(n_total * CFG["target_frac"])

idx = list(range(n_total))
random.shuffle(idx)

target_idx = idx[:n_target]

target_ds = Subset(dataset, target_idx)

print("Training target model...")

model, history = train_model(
    target_ds,
    CFG["target_epochs"],
    CFG,
    device,
    model_name="target"
)

torch.save(model.state_dict(), "target_model.pth")

print("Saved target_model.pth")