import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Load data
df = pd.read_csv("std_results_labeled.csv")

# Ratio Features
df["mid_low_ratio"] = (
    df["mid_mean"] /
    (df["low_mean"] + 1e-8)
)

df["high_mid_ratio"] = (
    df["high_mean"] /
    (df["mid_mean"] + 1e-8)
)

# Features
X = df[
    [
        "low_mean",
        "mid_mean",
        "high_mean",
        "mid_low_ratio",
        "high_mid_ratio"
    ]
]

y = df["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Model
clf = LogisticRegression(
    max_iter=5000
)

clf.fit(X_train, y_train)

# Predictions
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:, 1]

# Results
print("\n=== RATIO MODEL ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("AUC:", roc_auc_score(y_test, prob))

print("\nWeights:")
for name, weight in zip(X.columns, clf.coef_[0]):
    print(name, "=", weight)