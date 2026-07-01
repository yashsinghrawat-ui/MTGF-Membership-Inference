import json
from torchvision.datasets import CIFAR10

with open("results/nonmember_idx.json") as f:
    nonmembers = json.load(f)

dataset = CIFAR10(root="./data", train=True)

for i in range(20):

    idx = nonmembers[i]

    img, label = dataset[idx]

    img.save(f"nonmember_{i}.png")

    print(
        f"Saved nonmember_{i}.png",
        f"Index={idx}",
        f"Label={label}"
    )