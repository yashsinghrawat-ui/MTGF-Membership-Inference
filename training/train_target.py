import json
import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

import config
from mia_code.model import ImprovedUNet
from mia_code.train import train_model


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dataset_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                config.DATASET["normalize_mean"],
                config.DATASET["normalize_std"],
            ),
        ]
    )


def load_cifar10(train: bool = True, download: bool = True):
    return datasets.CIFAR10(
        root=str(config.DATASET["root"]),
        train=train,
        download=download,
        transform=dataset_transform(),
    )


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f)


def build_target_split() -> Tuple[list, list]:
    cfg = config.TRAINING
    rng = random.Random(config.SEED)
    idx = list(range(cfg["n_total"]))
    rng.shuffle(idx)

    n_target = int(cfg["n_total"] * cfg["target_frac"])
    target_idx = idx[:n_target]
    nonmember_idx = idx[n_target:]

    save_json(target_idx, config.PATHS["target_indices"])
    save_json(nonmember_idx, config.PATHS["nonmember_indices"])
    return target_idx, nonmember_idx


def load_target_model(checkpoint_path: Path | None = None) -> torch.nn.Module:
    device = select_device()
    ckpt = checkpoint_path or config.PATHS["target_checkpoint"]
    model = ImprovedUNet().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def train_or_load_target(force_train: bool = False) -> Path:
    config.ensure_directories()
    checkpoint = config.PATHS["target_checkpoint"]

    if checkpoint.exists() and not force_train:
        if (
            not config.PATHS["target_indices"].exists()
            or not config.PATHS["nonmember_indices"].exists()
        ):
            build_target_split()
        print(f"[target] checkpoint exists: {checkpoint}")
        return checkpoint

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    device = select_device()
    print(f"[target] device: {device}")

    dataset = load_cifar10(train=True, download=True)
    target_idx, _ = build_target_split()
    target_ds = Subset(dataset, target_idx)

    model, _ = train_model(
        target_ds,
        config.TRAINING["target_epochs"],
        config.legacy_training_cfg(),
        device,
        model_name="target",
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"[target] saved checkpoint: {checkpoint}")
    return checkpoint


def main() -> None:
    train_or_load_target(force_train=False)


if __name__ == "__main__":
    main()
