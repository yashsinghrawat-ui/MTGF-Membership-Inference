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