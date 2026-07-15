import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import random
from pathlib import Path

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

Path("shadows").mkdir(exist_ok=True)

for i in range(CFG["n_shadow"]):

    print(f"\n========== Shadow {i+1}/{CFG['n_shadow']} ==========")

    shadow_idx = random.sample(
        range(n_total),
        int(n_total * CFG["shadow_frac"])
    )

    shadow_ds = Subset(dataset, shadow_idx)

    shadow_model, _ = train_model(
        shadow_ds,
        CFG["shadow_epochs"],
        CFG,
        device,
        model_name=f"shadow-{i+1}"
    )

    torch.save(
        shadow_model.state_dict(),
        f"shadows/shadow_model_{i+1}.pth"
    )

    print(f"Saved shadows/shadow_model_{i+1}.pth")

print("\nAll shadow models completed.")