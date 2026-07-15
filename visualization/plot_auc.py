import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("research_results_v2.csv")

df["AUC"] = df["AUC"] * 100

colors = ["gray", "royalblue", "orange", "green"]

plt.figure(figsize=(8,5))

bars = plt.bar(df["Method"], df["AUC"], color=colors)

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2,
             h+0.2,
             f"{h:.2f}",
             ha="center",
             fontsize=10)

plt.ylabel("AUC (%)")
plt.xlabel("Methods")
plt.title("AUC Comparison")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig("Figure4_AUC_Comparison.png", dpi=600)

plt.show()