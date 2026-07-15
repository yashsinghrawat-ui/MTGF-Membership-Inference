import random
from pathlib import Path

import torch
from torch.utils.data import Subset

import config
from mia_code.train import train_model
from training.train_target import load_cifar10, select_device


def shadow_checkpoint(index: int) -> Path:
    return config.PATHS["shadow_dir"] / f"shadow_model_{index}.pth"


def existing_shadow_checkpoints() -> list[Path]:
    return [shadow_checkpoint(i) for i in range(1, config.TRAINING["n_shadow"] + 1)]


def train_or_load_shadows(force_train: bool = False) -> list[Path]:
    config.ensure_directories()
    checkpoints = existing_shadow_checkpoints()

    if all(path.exists() for path in checkpoints) and not force_train:
        print("[shadow] all checkpoints exist")
        return checkpoints

    device = select_device()
    dataset = load_cifar10(train=True, download=True)
    cfg = config.legacy_training_cfg()
    n_total = config.TRAINING["n_total"]
    n_split = int(n_total * config.TRAINING["shadow_frac"])

    saved = []
    for shadow_id in range(1, config.TRAINING["n_shadow"] + 1):
        ckpt = shadow_checkpoint(shadow_id)
        if ckpt.exists() and not force_train:
            print(f"[shadow-{shadow_id}] checkpoint exists: {ckpt}")
            saved.append(ckpt)
            continue

        seed = config.SEED + shadow_id * 1000
        rng = random.Random(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        shadow_idx = sorted(rng.sample(range(n_total), n_split))
        shadow_ds = Subset(dataset, shadow_idx)

        print(f"[shadow-{shadow_id}] training on {len(shadow_idx)} samples")
        model, _ = train_model(
            shadow_ds,
            config.TRAINING["shadow_epochs"],
            cfg,
            device,
            model_name=f"shadow-{shadow_id}",
        )

        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"[shadow-{shadow_id}] saved checkpoint: {ckpt}")
        saved.append(ckpt)

    return saved


def main() -> None:
    train_or_load_shadows(force_train=False)


if __name__ == "__main__":
    main()
