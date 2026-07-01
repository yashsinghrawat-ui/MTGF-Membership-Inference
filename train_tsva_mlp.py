import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("std_results_labeled.csv")

X = df.drop(columns=["image", "label"])
y = df["label"]

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
)

clf.fit(X_train, y_train)

pred_prob = clf.predict_proba(X_test)[:, 1]
pred = clf.predict(X_test)

print("\n====================")
print("MLP RESULTS")
print("====================")
print("Accuracy:", accuracy_score(y_test, pred))
print("AUC:", roc_auc_score(y_test, pred_prob))
print("====================")