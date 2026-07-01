import argparse
import csv
from pathlib import Path

import torch

from extract_research_features import load_image, load_model
from mia_code.leakage_features import (
    DEFAULT_N_MC,
    TIMESTEPS,
    BANDS,
    STATS,
    LeakageFeatureExtractor,
    feature_names,
)


def append_or_write_row(output_path: Path, image_path: str, features: dict) -> None:
    header = ["image", *feature_names()]

    if output_path.exists():
        with output_path.open("r", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != header:
            raise RuntimeError(
                f"{output_path} exists with a different schema. "
                "Move it aside or use a fresh output file."
            )

    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if output_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({"image": image_path, **features})


def print_summary(image_path: str, features: dict) -> None:
    print("\n===================================")
    print("Image           :", image_path)
    print("Raw Loss        :", f"{features['raw_loss']:.6f}")

    for timestep in TIMESTEPS:
        print(
            f"T={timestep}",
            f"RawMean={features[f'raw_loss_mean_{timestep}']:.6f}",
            f"RawStd={features[f'raw_loss_std_{timestep}']:.6f}",
            *[
                f"{band.capitalize()}{stat.capitalize()}="
                f"{features[f'{band}_{stat}_{timestep}']:.6f}"
                for stat in STATS
                for band in BANDS
            ],
        )

    print("===================================\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", default="std_results.csv")
    parser.add_argument("--n-mc", type=int, default=DEFAULT_N_MC)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--config", default="results/config.json")
    parser.add_argument("--checkpoint", default="target_model.pth")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alpha_bar = load_model(args.config, args.checkpoint, device)
    extractor = LeakageFeatureExtractor(
        model=model,
        alpha_bar=alpha_bar,
        device=device,
        n_mc=args.n_mc,
        base_seed=args.seed,
    )

    x = load_image(args.image, device)
    features = extractor.extract(x, image_id=args.image)

    if not args.no_save:
        append_or_write_row(Path(args.output), args.image, features)

    print_summary(args.image, features)


if __name__ == "__main__":
    main()
