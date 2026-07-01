from torchvision.datasets import CIFAR10

dataset = CIFAR10(
    root="./data",
    train=False,   # Test set = Non-members
    download=False
)

for i in range(500):

    img, label = dataset[i]

    img.save(f"nonmember_{i}.png")

    print(
        f"Saved nonmember_{i}.png",
        f"Label={label}"
    )