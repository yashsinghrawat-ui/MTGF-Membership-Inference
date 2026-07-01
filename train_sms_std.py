import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load CSV
df = pd.read_csv("std_results_labeled.csv")

# Features (Mean Only)
X = df[
    [
        "low_mean",
        "mid_mean",
        "high_mean"
    ]
]

# Labels
y = 1 - df["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# StandardScaler + Logistic Regression
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

# Train
clf.fit(X_train, y_train)

# Predict
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:, 1]

# Results
print("\n========================")
print("SCALED MEAN MODEL")
print("========================")
print("Accuracy:", accuracy_score(y_test, pred))
print("AUC:", roc_auc_score(y_test, prob))
print("========================\n")