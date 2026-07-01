import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("std_results_labeled.csv")

X = df.drop(columns=["image", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(
    max_iter=10000
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:, 1]

print("\n========================")
print("TSVA RESULTS")
print("========================")
print("Accuracy:", accuracy_score(y_test, pred))
print("AUC:", roc_auc_score(y_test, prob))
print("========================")

print("\nTop Features:")

weights = pd.Series(
    clf.coef_[0],
    index=X.columns
)

print(
    weights.abs()
    .sort_values(ascending=False)
    .head(15)
)