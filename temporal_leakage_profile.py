import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TIMESTEPS = [50, 100, 150, 200, 300, 400, 500, 750]


def cohens_d(member_values: np.ndarray, nonmember_values: np.ndarray) -> float:
    n_member = len(member_values)
    n_nonmember = len(nonmember_values)

    member_var = np.var(member_values, ddof=1)
    nonmember_var = np.var(nonmember_values, ddof=1)

    pooled_std = np.sqrt(
        ((n_member - 1) * member_var + (n_nonmember - 1) * nonmember_var)
        / (n_member + n_nonmember - 2)
    )

    if pooled_std == 0:
        return 0.0

    return float((np.mean(member_values) - np.mean(nonmember_values)) / pooled_std)


def compute_temporal_profile(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column with 1=member and 0=non-member.")

    rows = []

    for timestep in TIMESTEPS:
        feature = f"raw_loss_mean_{timestep}"
        if feature not in df.columns:
            raise ValueError(f"Missing required feature column: {feature}")

        member_values = df.loc[df["label"] == 1, feature].to_numpy(dtype=np.float64)
        nonmember_values = df.loc[df["label"] == 0, feature].to_numpy(dtype=np.float64)

        if len(member_values) < 2 or len(nonmember_values) < 2:
            raise ValueError(
                f"Need at least two samples per class for timestep {timestep}."
            )

        t_stat, p_value = stats.ttest_ind(
            member_values,
            nonmember_values,
            equal_var=False,
        )

        member_mean = float(np.mean(member_values))
        nonmember_mean = float(np.mean(nonmember_values))
        member_std = float(np.std(member_values, ddof=1))
        nonmember_std = float(np.std(nonmember_values, ddof=1))

        rows.append(
            {
                "Timestep": timestep,
                "Feature": feature,
                "Member_Mean": member_mean,
                "Nonmember_Mean": nonmember_mean,
                "Mean_Difference": member_mean - nonmember_mean,
                "Member_Std": member_std,
                "Nonmember_Std": nonmember_std,
                "Cohens_d": cohens_d(member_values, nonmember_values),
                "Welch_t_statistic": float(t_stat),
                "Welch_p_value": float(p_value),
                "N_Members": int(len(member_values)),
                "N_Nonmembers": int(len(nonmember_values)),
            }
        )

    return pd.DataFrame(rows)


def plot_temporal_profile(profile: pd.DataFrame, output_path: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    timesteps = profile["Timestep"].to_numpy()
    effect = profile["Cohens_d"].to_numpy()

    ax.plot(
        timesteps,
        effect,
        marker="o",
        linewidth=2.2,
        markersize=6,
        color="#2F6F8F",
        label="Raw denoising loss",
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")

    ax.set_xlabel("Diffusion timestep")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Temporal Leakage Profile")
    ax.set_xticks(timesteps)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="std_results_labeled.csv")
    parser.add_argument("--csv-out", default="temporal_leakage_profile.csv")
    parser.add_argument("--fig-out", default="Figure7_TemporalLeakageProfile.png")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    profile = compute_temporal_profile(df)

    profile.to_csv(args.csv_out, index=False)
    plot_temporal_profile(profile, args.fig_out)

    print(profile.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved temporal leakage profile: {args.csv_out}")
    print(f"Saved figure: {args.fig_out}")


if __name__ == "__main__":
    main()
