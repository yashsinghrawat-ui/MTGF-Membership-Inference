from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_bar(
    results_csv: Path,
    output_path: Path,
    metric: str = "AUC",
    title: str = "MTGF Membership Inference Performance",
) -> Path:
    df = pd.read_csv(results_csv)
    if metric not in df.columns:
        raise ValueError(f"Metric column not found: {metric}")

    err_col = f"{metric}_Std"
    yerr = df[err_col] if err_col in df.columns else None

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(
        df["Method"],
        df[metric],
        yerr=yerr,
        capsize=4,
        color="#4C78A8",
        edgecolor="#222222",
        linewidth=0.6,
    )
    if metric == "AUC":
        ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=15, ha="right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
