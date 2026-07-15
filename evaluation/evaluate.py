from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from evaluation.metrics import compute_metrics
from temporal_gradient_experiment import tgf_columns


RESULT_COLUMNS = [
    "Method",
    "Accuracy",
    "Accuracy_Std",
    "Balanced_Accuracy",
    "Balanced_Accuracy_Std",
    "AUC",
    "AUC_Std",
    "TPR_1FPR",
    "TPR_1FPR_Std",
    "TPR_5FPR",
    "TPR_5FPR_Std",
]


def logistic_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=config.CLASSIFIER["max_iter"],
            class_weight=config.CLASSIFIER["class_weight"],
            random_state=seed,
        ),
    )


def summarize(method: str, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    row: Dict[str, float] = {"Method": method}
    for metric in ("Accuracy", "Balanced_Accuracy", "AUC", "TPR_1FPR", "TPR_5FPR"):
        values = np.asarray([m[metric] for m in fold_metrics], dtype=np.float64)
        row[metric] = float(values.mean())
        row[f"{metric}_Std"] = float(values.std(ddof=1))
    return row


def evaluate_mtgf(force: bool = False) -> Path:
    output = config.PATHS["metrics_csv"]
    if output.exists() and not force:
        print(f"[eval] metrics CSV exists: {output}")
        return output

    labeled = pd.read_csv(config.PATHS["labeled_feature_csv"])
    mtgf = pd.read_csv(config.PATHS["mtgf_csv"])
    df = labeled[["image", "label"]].merge(mtgf, on=["image", "label"], how="inner")
    if len(df) != len(labeled):
        raise ValueError("MTGF feature merge changed row count.")

    features = tgf_columns()
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing MTGF columns: {missing}")

    fold_metrics = []
    for seed in config.EVALUATION["split_seeds"]:
        train_df, test_df = train_test_split(
            df,
            test_size=config.EVALUATION["test_size"],
            random_state=seed,
            stratify=df["label"],
        )
        y_train = train_df["label"].to_numpy(dtype=int)
        y_test = test_df["label"].to_numpy(dtype=int)

        clf = logistic_model(seed)
        clf.fit(train_df[features], y_train)
        scores = clf.predict_proba(test_df[features])[:, 1]
        predictions = clf.predict(test_df[features])
        fold_metrics.append(compute_metrics(y_test, scores, predictions))

    results = pd.DataFrame([summarize("MTGF + Logistic Regression", fold_metrics)])
    output.parent.mkdir(parents=True, exist_ok=True)
    results[RESULT_COLUMNS].to_csv(output, index=False)
    print(f"[eval] saved metrics: {output}")
    print(results[RESULT_COLUMNS].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return output


def main() -> None:
    evaluate_mtgf(force=False)


if __name__ == "__main__":
    main()
