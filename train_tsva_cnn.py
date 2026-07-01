import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# --------------------------
# Load Data
# --------------------------

df = pd.read_csv("std_results_labeled.csv")

X = df.drop(columns=["image", "label"]).values
y = df["label"].values

# --------------------------
# Scale Features
# --------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)

# --------------------------
# Reshape
# 24 features -> 4x6x1
# --------------------------

X = X.reshape(-1, 4, 6, 1)

# --------------------------
# Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# --------------------------
# CNN
# --------------------------

model = Sequential([
    Conv2D(
        16,
        (2,2),
        activation="relu",
        input_shape=(4,6,1)
    ),

    Flatten(),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------
# Train
# --------------------------

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    verbose=1
)

# --------------------------
# Evaluate
# --------------------------

prob = model.predict(X_test).flatten()

pred = (prob > 0.5).astype(int)

print("\n========================")
print("TSVA-CNN RESULTS")
print("========================")
print("Accuracy:", accuracy_score(y_test, pred))
print("AUC:", roc_auc_score(y_test, prob))
print("========================")