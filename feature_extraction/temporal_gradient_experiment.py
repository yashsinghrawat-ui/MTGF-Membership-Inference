import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TIMESTEPS = [50, 100, 150, 200, 300, 400, 500, 750]
ADJACENT_PAIRS = list(zip(TIMESTEPS[:-1], TIMESTEPS[1:]))
BANDS = ("low", "mid", "high")
STATS = ("mean", "std")
SPLIT_SEEDS = [11, 22, 33, 44, 55]

HYBRID_TOP_TSVA_FEATURES = [
    "high_mean_500",
    "mid_mean_400",
    "mid_mean_50",
    "high_mean_300",
    "mid_mean_500",
    "mid_mean_100",
    "high_mean_750",
    "low_mean_300",
    "high_mean_150",
    "high_mean_400",
]

RESULT_COLUMNS = [
    "Method",
    "Accuracy",
    "Accuracy_Std",
    "Balanced_Accuracy",
    "Balanced_Accuracy_Std",
    "AUC",
    "AUC_Std",
    "TPR_1FPR",
    "TPR_1FPR_Std",
    "TPR_5FPR",
    "TPR_5FPR_Std",
]


def tsva_v2_columns() -> List[str]:
    return [
        f"{band}_{stat}_{timestep}"
        for timestep in TIMESTEPS
        for stat in STATS
        for band in BANDS
    ]


def tgf_source_prefixes() -> List[str]:
    return [
        f"{band}_{stat}"
        for stat in STATS
        for band in BANDS
    ]


def tgf_columns() -> List[str]:
    columns = []
    for t0, t1 in ADJACENT_PAIRS:
        for prefix in tgf_source_prefixes():
            columns.append(f"delta_{prefix}_{t0}_{t1}")
    return columns


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def refuse_to_overwrite(paths: Iterable[str], overwrite: bool) -> None:
    if overwrite:
        return

    existing = [path for path in paths if Path(path).exists()]
    if existing:
        joined = ", ".join(existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output file(s): {joined}. "
            "Move them aside or re-run with --overwrite."
        )


def build_temporal_gradient_features(df: pd.DataFrame) -> pd.DataFrame:
    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected input CSV to contain 'image' and 'label' columns.")

    output = df[["image", "label"]].copy()

    for t0, t1 in ADJACENT_PAIRS:
        for prefix in tgf_source_prefixes():
            col_t0 = f"{prefix}_{t0}"
            col_t1 = f"{prefix}_{t1}"
            delta_col = f"delta_{prefix}_{t0}_{t1}"

            require_columns(df, [col_t0, col_t1], f"TGF pair {t0}->{t1}")
            output[delta_col] = df[col_t1] - df[col_t0]

    return output


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = tpr[fpr <= target_fpr]
    return float(valid.max()) if len(valid) else 0.0


def compute_metrics(
    y_test: np.ndarray,
    scores: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, scores),
        "tpr_1fpr": tpr_at_fpr(y_test, scores, 0.01),
        "tpr_5fpr": tpr_at_fpr(y_test, scores, 0.05),
    }


def summarize(method: str, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    row: Dict[str, float] = {"Method": method}
    metric_map = {
        "Accuracy": "accuracy",
        "Balanced_Accuracy": "balanced_accuracy",
        "AUC": "auc",
        "TPR_1FPR": "tpr_1fpr",
        "TPR_5FPR": "tpr_5fpr",
    }

    for out_name, key in metric_map.items():
        values = np.asarray([m[key] for m in fold_metrics], dtype=np.float64)
        row[out_name] = float(values.mean())
        row[f"{out_name}_Std"] = float(values.std(ddof=1))

    return row


def logistic_model(seed: int):
    # Exact settings used in evaluate_research_methods.py.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def evaluate_logistic_group(
    df: pd.DataFrame,
    columns: List[str],
    method: str,
) -> Dict[str, float]:
    require_columns(df, columns, method)
    fold_metrics = []

    for seed in SPLIT_SEEDS:
        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=seed,
            stratify=df["label"],
        )

        y_train = train_df["label"].to_numpy(dtype=int)
        y_test = test_df["label"].to_numpy(dtype=int)

        clf = logistic_model(seed)
        clf.fit(train_df[columns], y_train)
        scores = clf.predict_proba(test_df[columns])[:, 1]
        pred = clf.predict(test_df[columns])
        fold_metrics.append(compute_metrics(y_test, scores, pred))

    return summarize(method, fold_metrics)


def best_threshold_from_train(y_train: np.ndarray, train_scores: np.ndarray) -> float:
    thresholds = np.unique(train_scores)
    if len(thresholds) > 1000:
        thresholds = np.quantile(train_scores, np.linspace(0.0, 1.0, 1000))

    best_threshold = thresholds[0]
    best_score = -np.inf
    for threshold in thresholds:
        pred = (train_scores >= threshold).astype(int)
        score = balanced_accuracy_score(y_train, pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def evaluate_raw_loss_score(df: pd.DataFrame) -> Dict[str, float]:
    require_columns(df, ["raw_loss"], "Raw Loss")
    fold_metrics = []

    for seed in SPLIT_SEEDS:
        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=seed,
            stratify=df["label"],
        )

        y_train = train_df["label"].to_numpy(dtype=int)
        y_test = test_df["label"].to_numpy(dtype=int)
        train_scores = -train_df["raw_loss"].to_numpy(dtype=np.float64)
        test_scores = -test_df["raw_loss"].to_numpy(dtype=np.float64)

        threshold = best_threshold_from_train(y_train, train_scores)
        pred = (test_scores >= threshold).astype(int)
        fold_metrics.append(compute_metrics(y_test, test_scores, pred))

    return summarize("Raw Loss", fold_metrics)


def evaluate_all_methods(full_df: pd.DataFrame, tgf_df: pd.DataFrame) -> pd.DataFrame:
    eval_df = full_df.merge(tgf_df, on=["image", "label"], how="inner")
    if len(eval_df) != len(full_df):
        raise ValueError("TGF merge changed row count; check image/label keys.")

    rows = [
        evaluate_raw_loss_score(eval_df),
        evaluate_logistic_group(eval_df, tsva_v2_columns(), "TSVA-v2"),
        evaluate_logistic_group(
            eval_df,
            ["raw_loss", *HYBRID_TOP_TSVA_FEATURES],
            "Hybrid TSVA + Raw Loss",
        ),
        evaluate_logistic_group(eval_df, tgf_columns(), "Temporal Gradient Features"),
    ]

    return pd.DataFrame(rows)[RESULT_COLUMNS]


def plot_comparison(results: pd.DataFrame, output_path: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    colors = ["#6E6E6E", "#4C78A8", "#F58518", "#54A24B"]
    labels = ["Raw Loss", "TSVA-v2", "Hybrid", "TGF"]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(results))
    ax.bar(
        x,
        results["AUC"],
        yerr=results["AUC_Std"],
        color=colors,
        capsize=4,
        edgecolor="#222222",
        linewidth=0.6,
    )

    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.45, max(0.70, float(results["AUC"].max() + 0.08)))
    ax.set_ylabel("AUC")
    ax.set_title("Temporal Gradient Feature Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="std_results_labeled.csv")
    parser.add_argument("--tgf-out", default="temporal_gradient_features.csv")
    parser.add_argument("--results-out", default="research_results_v2.csv")
    parser.add_argument(
        "--figure-out",
        default="Figure8_TemporalGradientComparison.png",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    refuse_to_overwrite(
        [args.tgf_out, args.results_out, args.figure_out],
        overwrite=args.overwrite,
    )

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    full_df = pd.read_csv(input_path)
    if "label" not in full_df.columns:
        raise ValueError("Expected input CSV to contain a 'label' column.")

    tgf_df = build_temporal_gradient_features(full_df)
    tgf_df.to_csv(args.tgf_out, index=False)

    results = evaluate_all_methods(full_df, tgf_df)
    results.to_csv(args.results_out, index=False)
    plot_comparison(results, args.figure_out)

    print("\n==============================")
    print("Temporal Gradient Feature Results")
    print("==============================")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved TGF features: {args.tgf_out}")
    print(f"Saved results: {args.results_out}")
    print(f"Saved figure: {args.figure_out}")


if __name__ == "__main__":
    main()
