import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Create output folder
# ----------------------------
os.makedirs("figures", exist_ok=True)

plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams["font.size"] = 11

# =====================================================
# Figure 1 : Main Comparison
# =====================================================

main = pd.read_csv("research_results.csv")

plt.figure()

plt.bar(main["Method"], main["AUC"])

plt.ylabel("AUC")
plt.xlabel("Method")
plt.title("Comparison of MIA Methods")
plt.xticks(rotation=20, ha="right")

plt.tight_layout()

plt.savefig(
    "figures/Figure1_MainComparison.png",
    dpi=300
)

plt.close()

print("✓ Figure1 saved")

# =====================================================
# Figure 2 : Temporal Localization
# =====================================================

temp = pd.read_csv("temporal_ablation_results.csv")

temp = temp[
    temp["Method"].str.contains(r"timestep_\d+_raw_tsva", regex=True)
].copy()

temp["timestep"] = pd.to_numeric(
    temp["Method"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

temp = temp.dropna(subset=["timestep"])

temp["timestep"] = temp["timestep"].astype(int)

temp = temp.sort_values("timestep")

plt.figure()

plt.plot(
    temp["timestep"],
    temp["AUC"],
    marker="o"
)

plt.xlabel("Diffusion Timestep")

plt.ylabel("AUC")

plt.title("Temporal Localization")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "figures/Figure2_Temporal.png",
    dpi=300
)

plt.close()

print("✓ Figure2 saved")

print("\nPart 1 Complete.")

print("✓ Figure2 saved")

print("\nPart 1 Complete.")

# =====================================================
# Figure 3 : Frequency Localization
# =====================================================

freq = pd.read_csv("frequency_ablation_results.csv")

freq = freq[
    freq["Method"].isin([
        "low_band_only",
        "mid_band_only",
        "high_band_only",
        "mid_high_bands",
        "all_bands"
    ])
]

plt.figure(figsize=(8,5))

plt.bar(freq["Method"], freq["AUC"])

plt.ylabel("AUC")
plt.xlabel("Frequency Band")
plt.title("Frequency Localization")

plt.xticks(rotation=20, ha="right")

plt.tight_layout()

plt.savefig(
    "figures/Figure3_Frequency.png",
    dpi=300
)

plt.close()

print("✓ Figure3 saved")

# ---------------- Figure 4 ----------------

importance = pd.read_csv("feature_importance_results.csv").head(10)

plt.figure(figsize=(8,5))

plt.barh(
    importance["Feature"][::-1],
    importance["MeanAbsWeight"][::-1]
)

plt.xlabel("Mean Absolute Weight")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig(
    "figures/Figure4_FeatureImportance.png",
    dpi=300
)

plt.close()

print("✓ Figure4 saved")

# ---------------- Figure 5 ----------------

corr = pd.read_csv("score_correlation_results.csv")

plt.figure(figsize=(8,5))

plt.bar(
    corr["ScoreB"],
    corr["PearsonR"]
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Pearson Correlation")
plt.title("Correlation Between Raw Loss and TSVA Features")

plt.tight_layout()

plt.savefig(
    "figures/Figure5_Correlation.png",
    dpi=300
)

plt.close()

print("✓ Figure5 saved")

# ---------------- Figure 6 ----------------

import numpy as np

importance = pd.read_csv("feature_importance_results.csv").head(10)

features = importance["Feature"].tolist()

heat = []

for f in features:
    values = []
    for t in [50,100,150,200,300,400,500,750]:
        if str(t) in f:
            values.append(importance.loc[
                importance["Feature"]==f,
                "MeanAbsWeight"
            ].values[0])
        else:
            values.append(0)
    heat.append(values)

plt.figure(figsize=(10,6))

plt.imshow(
    heat,
    aspect="auto"
)

plt.colorbar(label="Importance")

plt.xticks(
    range(8),
    [50,100,150,200,300,400,500,750]
)

plt.yticks(
    range(len(features)),
    features
)

plt.xlabel("Diffusion Timestep")
plt.ylabel("Feature")

plt.title("Temporal Feature Importance Heatmap")

plt.tight_layout()

plt.savefig(
    "figures/Figure6_Heatmap.png",
    dpi=300
)

plt.close()

print("✓ Figure6 saved")