import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

df = pd.read_csv("std_results_labeled.csv")


X = df[
    [
        "low_mean_50","mid_mean_50","high_mean_50",
        "low_std_50","mid_std_50","high_std_50",

        "low_mean_100","mid_mean_100","high_mean_100",
        "low_std_100","mid_std_100","high_std_100",

        "low_mean_150","mid_mean_150","high_mean_150",
        "low_std_150","mid_std_150","high_std_150",

        "low_mean_200","mid_mean_200","high_mean_200",
        "low_std_200","mid_std_200","high_std_200"
    ]
]

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
prob = clf.predict_proba(X_test)[:,1]

print("\n========================")
print("MT-SMS RESULTS")
print("========================")
print("Accuracy:", accuracy_score(y_test, pred))
print("AUC:", roc_auc_score(y_test, prob))
print("========================")

print("\nWeights:")

for name, w in zip(X.columns, clf.coef_[0]):
    print(name, "=", w)