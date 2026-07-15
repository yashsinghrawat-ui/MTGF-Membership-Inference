from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = tpr[fpr <= target_fpr]
    return float(valid.max()) if len(valid) else 0.0


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(y_true, scores),
        "TPR_1FPR": tpr_at_fpr(y_true, scores, 0.01),
        "TPR_5FPR": tpr_at_fpr(y_true, scores, 0.05),
    }
