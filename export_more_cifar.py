from torchvision.datasets import CIFAR10

dataset = CIFAR10(
    root="./data",
    train=True,
    download=False
)

for i in range(50):
    img, label = dataset[i]

    img.save(f"cifar_{i}.png")

    print(
        f"Saved cifar_{i}.png",
        f"Label={label}"
    )