from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


SEED = 42
DEVICE = "auto"


DATASET = {
    "name": "CIFAR10",
    "root": ROOT_DIR / "data",
    "train": True,
    "download": True,
    "image_size": 32,
    "normalize_mean": [0.5, 0.5, 0.5],
    "normalize_std": [0.5, 0.5, 0.5],
}


TRAINING = {
    "n_total": 5000,
    "target_frac": 0.5,
    "shadow_frac": 0.5,
    "n_shadow": 5,
    "target_epochs": 50,
    "shadow_epochs": 25,
    "batch_size": 16,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "print_every": 20,
    "T": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}


EVALUATION = {
    "n_members": 500,
    "n_nonmembers": 500,
    "test_size": 0.30,
    "split_seeds": [11, 22, 33, 44, 55],
}


FEATURES = {
    "timesteps": [50, 100, 150, 200, 300, 400, 500, 750],
    "adjacent_pairs": [
        (50, 100),
        (100, 150),
        (150, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, 750),
    ],
    "bands": ["low", "mid", "high"],
    "stats": ["mean", "std"],
    "n_mc": 50,
    "raw_loss_timestep": 100,
}


CLASSIFIER = {
    "name": "LogisticRegression",
    "max_iter": 10000,
    "class_weight": "balanced",
}


PATHS = {
    "models_dir": ROOT_DIR / "models",
    "checkpoints_dir": ROOT_DIR / "checkpoints",
    "graphs_dir": ROOT_DIR / "graphs",
    "logs_dir": ROOT_DIR / "logs",
    "results_dir": ROOT_DIR / "results",
    "target_checkpoint": ROOT_DIR / "checkpoints" / "target_model.pth",
    "shadow_dir": ROOT_DIR / "checkpoints" / "shadows",
    "target_indices": ROOT_DIR / "results" / "target_idx.json",
    "nonmember_indices": ROOT_DIR / "results" / "nonmember_idx.json",
    "member_dir": ROOT_DIR / "members",
    "feature_csv": ROOT_DIR / "results" / "std_results.csv",
    "labeled_feature_csv": ROOT_DIR / "results" / "std_results_labeled.csv",
    "mtgf_csv": ROOT_DIR / "results" / "temporal_gradient_features.csv",
    "metrics_csv": ROOT_DIR / "results" / "research_results_v2.csv",
    "figure8": ROOT_DIR / "graphs" / "Figure8_TemporalGradientComparison.png",
}


def legacy_training_cfg() -> dict:
    cfg = dict(TRAINING)
    cfg.update(
        {
            "n_eval": EVALUATION["n_members"],
            "n_nonmember": EVALUATION["n_nonmembers"],
            "t_sweep": FEATURES["timesteps"],
            "t_lira": FEATURES["raw_loss_timestep"],
            "t_strong": [50, 100, 150, 200],
            "n_repeats": FEATURES["n_mc"],
            "n_probe": EVALUATION["n_members"] + EVALUATION["n_nonmembers"],
            "seed": SEED,
        }
    )
    return cfg


def ensure_directories() -> None:
    for key in (
        "models_dir",
        "checkpoints_dir",
        "graphs_dir",
        "logs_dir",
        "results_dir",
    ):
        PATHS[key].mkdir(parents=True, exist_ok=True)
    PATHS["shadow_dir"].mkdir(parents=True, exist_ok=True)
    PATHS["member_dir"].mkdir(parents=True, exist_ok=True)
