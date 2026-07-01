import json
from torchvision.datasets import CIFAR10

# Load member indices
with open("results/target_sample_idx.json", "r") as f:
    member_idx = json.load(f)

print("Total members:", len(member_idx))

# CIFAR10 load
dataset = CIFAR10(
    root="./data",
    train=True,
    download=True
)

# First member image
idx = member_idx[0]

img, label = dataset[idx]

img.save("member_image.png")

print("Saved member image")
print("Index :", idx)
print("Label :", label)