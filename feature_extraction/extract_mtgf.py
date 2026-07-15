import csv
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import CIFAR10

import config
from mia_code.leakage_features import LeakageFeatureExtractor, feature_names
from mia_code.model import ImprovedUNet, make_schedule
from temporal_gradient_experiment import build_temporal_gradient_features


def image_transform():
    return T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(
                config.DATASET["normalize_mean"],
                config.DATASET["normalize_std"],
            ),
        ]
    )


def load_json(path: Path) -> list:
    with path.open("r") as f:
        return json.load(f)


def ensure_challenge_images() -> None:
    member_dir = config.PATHS["member_dir"]
    member_dir.mkdir(parents=True, exist_ok=True)

    n_members = config.EVALUATION["n_members"]
    n_nonmembers = config.EVALUATION["n_nonmembers"]
    member_done = all((member_dir / f"member_{i}.png").exists() for i in range(n_members))
    nonmember_done = all((config.ROOT_DIR / f"nonmember_{i}.png").exists() for i in range(n_nonmembers))

    if member_done and nonmember_done:
        print("[features] challenge images already exist")
        return

    if not config.PATHS["target_indices"].exists() or not config.PATHS["nonmember_indices"].exists():
        raise FileNotFoundError(
            "Target/nonmember index files are missing. Train the target model first."
        )

    dataset = CIFAR10(
        root=str(config.DATASET["root"]),
        train=True,
        download=True,
    )
    target_idx = load_json(config.PATHS["target_indices"])
    nonmember_idx = load_json(config.PATHS["nonmember_indices"])

    for i, idx in enumerate(target_idx[:n_members]):
        out = member_dir / f"member_{i}.png"
        if not out.exists():
            img, _ = dataset[idx]
            img.save(out)

    for i, idx in enumerate(nonmember_idx[:n_nonmembers]):
        out = config.ROOT_DIR / f"nonmember_{i}.png"
        if not out.exists():
            img, _ = dataset[idx]
            img.save(out)

    print("[features] challenge images ready")


def build_image_list() -> List[str]:
    return [
        *[f"members/member_{i}.png" for i in range(config.EVALUATION["n_members"])],
        *[f"nonmember_{i}.png" for i in range(config.EVALUATION["n_nonmembers"])],
    ]


def validate_paths(paths: Iterable[str]) -> None:
    missing = [p for p in paths if not (config.ROOT_DIR / p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing challenge images: {missing[:5]}")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_target(device: torch.device):
    checkpoint = config.PATHS["target_checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Target checkpoint not found: {checkpoint}")

    model = ImprovedUNet().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cfg = config.legacy_training_cfg()
    alpha_bar = make_schedule(
        cfg["T"],
        cfg["beta_start"],
        cfg["beta_end"],
        device=device,
    )
    return model, alpha_bar


def load_image(path: str, transform, device: torch.device) -> torch.Tensor:
    img = Image.open(config.ROOT_DIR / path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def extract_multi_timestep_reconstruction_loss(force: bool = False) -> Path:
    output = config.PATHS["feature_csv"]
    if output.exists() and not force:
        print(f"[features] feature CSV exists: {output}")
        return output

    ensure_challenge_images()
    paths = build_image_list()
    validate_paths(paths)

    device = select_device()
    model, alpha_bar = load_target(device)
    extractor = LeakageFeatureExtractor(
        model=model,
        alpha_bar=alpha_bar,
        device=device,
        n_mc=config.FEATURES["n_mc"],
        base_seed=config.SEED,
    )
    transform = image_transform()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", *feature_names()])
        writer.writeheader()
        for idx, image_path in enumerate(paths, start=1):
            x = load_image(image_path, transform, device)
            features = extractor.extract(x, image_id=image_path)
            writer.writerow({"image": image_path, **features})
            if idx == 1 or idx % 25 == 0 or idx == len(paths):
                print(f"[features] {idx:04d}/{len(paths):04d} {image_path}")

    print(f"[features] saved: {output}")
    return output


def infer_label(image_path: str) -> int:
    if "member_" in image_path and "nonmember_" not in image_path:
        return 1
    if "nonmember_" in image_path:
        return 0
    raise ValueError(f"Cannot infer membership label from path: {image_path}")


def label_feature_csv(force: bool = False) -> Path:
    output = config.PATHS["labeled_feature_csv"]
    if output.exists() and not force:
        print(f"[features] labeled CSV exists: {output}")
        return output

    df = pd.read_csv(config.PATHS["feature_csv"])
    df["label"] = df["image"].apply(infer_label)
    counts = df["label"].value_counts().to_dict()
    expected = {
        1: config.EVALUATION["n_members"],
        0: config.EVALUATION["n_nonmembers"],
    }
    if counts != expected:
        raise ValueError(f"Expected labels {expected}, found {counts}")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"[features] saved labeled CSV: {output}")
    return output


def extract_mtgf(force: bool = False) -> Path:
    output = config.PATHS["mtgf_csv"]
    if output.exists() and not force:
        print(f"[mtgf] CSV exists: {output}")
        return output

    df = pd.read_csv(config.PATHS["labeled_feature_csv"])
    mtgf = build_temporal_gradient_features(df)
    output.parent.mkdir(parents=True, exist_ok=True)
    mtgf.to_csv(output, index=False)
    print(f"[mtgf] saved: {output}")
    return output


def run_feature_pipeline(force: bool = False) -> tuple[Path, Path, Path]:
    feature_csv = extract_multi_timestep_reconstruction_loss(force=force)
    labeled_csv = label_feature_csv(force=force)
    mtgf_csv = extract_mtgf(force=force)
    return feature_csv, labeled_csv, mtgf_csv


def main() -> None:
    run_feature_pipeline(force=False)


if __name__ == "__main__":
    main()
