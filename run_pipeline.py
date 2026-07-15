import importlib
import sys

import config


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "PIL",
]


def detect_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        print(f"[env] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[env] Apple MPS detected")
    else:
        print("[env] GPU not detected; running on CPU")


def check_dependencies() -> None:
    missing = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)

    if missing:
        raise RuntimeError(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + ". Install them with: pip install -r requirements.txt"
        )
    print("[env] dependency check passed")


def check_dataset() -> None:
    from torchvision.datasets import CIFAR10

    CIFAR10(root=str(config.DATASET["root"]), train=True, download=True)
    CIFAR10(root=str(config.DATASET["root"]), train=False, download=True)
    print("[data] CIFAR-10 is available")


def main() -> None:
    print("=" * 72)
    print("SMS-LiRA / MTGF Reproducible Research Pipeline")
    print("=" * 72)

    detect_gpu()
    check_dependencies()
    config.ensure_directories()
    check_dataset()

    print("\n[1/5] Target model")
    from training.train_target import train_or_load_target

    train_or_load_target(force_train=False)

    print("\n[2/5] Shadow models")
    from training.train_shadow import train_or_load_shadows

    train_or_load_shadows(force_train=False)

    print("\n[3/5] Multi-timestep reconstruction loss + MTGF extraction")
    from feature_extraction.extract_mtgf import run_feature_pipeline

    run_feature_pipeline(force=False)

    print("\n[4/5] Logistic Regression evaluation")
    from evaluation.evaluate import evaluate_mtgf

    evaluate_mtgf(force=False)

    print("\n[5/5] Figures")
    from visualization.generate_figures import generate_figures

    generate_figures(force=False)

    print("\nPipeline complete.")
    print(f"Results: {config.PATHS['metrics_csv']}")
    print(f"MTGF features: {config.PATHS['mtgf_csv']}")
    print(f"Figure: {config.PATHS['figure8']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        raise
