import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, List

import torch
import torchvision.transforms as T
from PIL import Image

from mia_code.leakage_features import (
    DEFAULT_N_MC,
    LeakageFeatureExtractor,
    feature_names,
)
from mia_code.model import ImprovedUNet, make_schedule


def build_image_list(n_per_class: int) -> List[str]:
    member_paths = [f"members/member_{i}.png" for i in range(n_per_class)]
    nonmember_paths = [f"nonmember_{i}.png" for i in range(n_per_class)]
    return member_paths + nonmember_paths


def validate_paths(paths: Iterable[str]) -> None:
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} image files. First missing files: {preview}"
        )


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = ImprovedUNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    alpha_bar = make_schedule(
        cfg["T"],
        cfg["beta_start"],
        cfg["beta_end"],
        device=device,
    )
    return model, alpha_bar


def load_image(path: str, device: torch.device) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def write_features(args) -> None:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output} already exists. Use --overwrite to regenerate it."
        )

    paths = build_image_list(args.n_per_class)
    validate_paths(paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading target diffusion model once...")

    model, alpha_bar = load_model(args.config, args.checkpoint, device)
    extractor = LeakageFeatureExtractor(
        model=model,
        alpha_bar=alpha_bar,
        device=device,
        n_mc=args.n_mc,
        base_seed=args.seed,
    )

    header = ["image", *feature_names()]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for idx, path in enumerate(paths, start=1):
            x = load_image(path, device)
            features = extractor.extract(x, image_id=path)
            writer.writerow({"image": path, **features})

            if idx == 1 or idx % args.print_every == 0 or idx == len(paths):
                print(f"[{idx:04d}/{len(paths):04d}] extracted {path}", flush=True)

    print(f"Saved features: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="std_results.csv")
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--n-mc", type=int, default=DEFAULT_N_MC)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--config", default="results/config.json")
    parser.add_argument("--checkpoint", default="target_model.pth")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    write_features(args)


if __name__ == "__main__":
    main()
