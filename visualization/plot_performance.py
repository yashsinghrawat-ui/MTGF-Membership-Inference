import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("research_results_v2.csv")

# Convert to percentage
df["Accuracy"] *= 100

# Create figure
plt.figure(figsize=(8,5))

bars = plt.bar(df["Method"], df["Accuracy"])

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.2,
        f"{height:.2f}%",
        ha="center",
        fontsize=10
    )

plt.title("Performance Comparison (Accuracy)")
plt.xlabel("Methods")
plt.ylabel("Accuracy (%)")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig("Figure3_Performance_Accuracy.png", dpi=600)

plt.show()

print("Figure saved as Figure3_Performance_Accuracy.png")