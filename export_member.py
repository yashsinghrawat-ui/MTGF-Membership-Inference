import json
from torchvision.datasets import CIFAR10
import os

os.makedirs("members", exist_ok=True)

with open("results/target_idx.json", "r") as f:
    member_idx = json.load(f)

dataset = CIFAR10(
    root="./data",
    train=True,
    download=False
)

for i, idx in enumerate(member_idx[:1000]):
    img, label = dataset[idx]
    img.save(f"members/member_{i}.png")

print("Saved 1000 member images")