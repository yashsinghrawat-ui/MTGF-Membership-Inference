import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mia_code.leakage_features import (
    BANDS,
    EARLY_TIMESTEPS,
    LATE_TIMESTEPS,
    MID_TIMESTEPS,
    STATS,
    TIMESTEPS,
    raw_columns_for_timesteps,
    raw_loss_columns,
    timestep_columns,
    tsva_columns_for_bands,
    tsva_columns_for_stats,
    tsva_columns_for_timesteps,
    tsva_v2_columns,
)


SPLIT_SEEDS = [11, 22, 33, 44, 55]
N_PER_CLASS = 500

HYBRID_TOP_TSVA_FEATURES = [
    "high_mean_500",
    "mid_mean_400",
    "mid_mean_50",
    "high_mean_300",
    "mid_mean_500",
    "mid_mean_100",
    "high_mean_750",
    "low_mean_300",
    "high_mean_150",
    "high_mean_400",
]

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


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_balanced_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["label"].isin([0, 1])].copy()
    counts = df["label"].value_counts().to_dict()
    expected = {0: N_PER_CLASS, 1: N_PER_CLASS}
    if counts != expected:
        raise ValueError(
            f"Expected exactly balanced paper data {expected}, but found {counts}."
        )
    return df.reset_index(drop=True)


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = tpr[fpr <= target_fpr]
    return float(valid.max()) if len(valid) else 0.0


def best_threshold_from_train(y_train: np.ndarray, train_scores: np.ndarray) -> float:
    thresholds = np.unique(train_scores)
    if len(thresholds) > 1000:
        thresholds = np.quantile(train_scores, np.linspace(0.0, 1.0, 1000))

    best_threshold = thresholds[0]
    best_score = -np.inf
    for threshold in thresholds:
        pred = (train_scores >= threshold).astype(int)
        score = balanced_accuracy_score(y_train, pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def compute_metrics(
    y_test: np.ndarray,
    scores: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, scores),
        "tpr_1fpr": tpr_at_fpr(y_test, scores, 0.01),
        "tpr_5fpr": tpr_at_fpr(y_test, scores, 0.05),
    }


def summarize_metrics(method: str, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    row: Dict[str, float] = {"Method": method}
    metric_map = {
        "Accuracy": "accuracy",
        "Balanced_Accuracy": "balanced_accuracy",
        "AUC": "auc",
        "TPR_1FPR": "tpr_1fpr",
        "TPR_5FPR": "tpr_5fpr",
    }
    for output_name, metric_name in metric_map.items():
        values = np.asarray([m[metric_name] for m in fold_metrics], dtype=np.float64)
        row[output_name] = float(values.mean())
        row[f"{output_name}_Std"] = float(values.std(ddof=1))
    return row


def logistic_model(random_state: int = 42):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )


def evaluate_logistic_group(
    df: pd.DataFrame,
    columns: List[str],
    seeds: Iterable[int] = SPLIT_SEEDS,
) -> Tuple[List[Dict[str, float]], List[pd.Series]]:
    require_columns(df, columns, "Logistic feature group")
    metrics = []
    weights = []

    for seed in seeds:
        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=seed,
            stratify=df["label"],
        )
        y_train = train_df["label"].to_numpy(dtype=int)
        y_test = test_df["label"].to_numpy(dtype=int)

        clf = logistic_model(random_state=seed)
        clf.fit(train_df[columns], y_train)
        scores = clf.predict_proba(test_df[columns])[:, 1]
        pred = clf.predict(test_df[columns])

        metrics.append(compute_metrics(y_test, scores, pred))
        weights.append(
            pd.Series(
                clf.named_steps["logisticregression"].coef_[0],
                index=columns,
            )
        )

    return metrics, weights


def evaluate_raw_score_group(
    df: pd.DataFrame,
    score_column: str,
    method_name: str,
    seeds: Iterable[int] = SPLIT_SEEDS,
) -> List[Dict[str, float]]:
    require_columns(df, [score_column], method_name)
    metrics = []

    for seed in seeds:
        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=seed,
            stratify=df["label"],
        )
        y_train = train_df["label"].to_numpy(dtype=int)
        y_test = test_df["label"].to_numpy(dtype=int)

        train_scores = -train_df[score_column].to_numpy(dtype=np.float64)
        test_scores = -test_df[score_column].to_numpy(dtype=np.float64)
        threshold = best_threshold_from_train(y_train, train_scores)
        pred = (test_scores >= threshold).astype(int)
        metrics.append(compute_metrics(y_test, test_scores, pred))

    return metrics


def gaussian_log_likelihood_ratio(
    train_x: np.ndarray,
    y_train: np.ndarray,
    eval_x: np.ndarray,
) -> np.ndarray:
    x_member = train_x[y_train == 1]
    x_nonmember = train_x[y_train == 0]

    mu_member = x_member.mean(axis=0)
    mu_nonmember = x_nonmember.mean(axis=0)
    var_member = np.maximum(x_member.var(axis=0), 1e-12)
    var_nonmember = np.maximum(x_nonmember.var(axis=0), 1e-12)

    member_ll = -0.5 * (
        np.log(var_member) + ((eval_x - mu_member) ** 2) / var_member
    )
    nonmember_ll = -0.5 * (
        np.log(var_nonmember) + ((eval_x - mu_nonmember) ** 2) / var_nonmember
    )
    return (member_ll - nonmember_ll).sum(axis=1)


def idea2_features_for_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    y_train = train_df["label"].to_numpy(dtype=int)
    out = pd.DataFrame(
        {
            "image": eval_df["image"].to_numpy(),
            "label": eval_df["label"].to_numpy(dtype=int),
        }
    )
    llr_columns = []

    for timestep in TIMESTEPS:
        columns = timestep_columns(timestep)
        require_columns(train_df, columns, f"Timestep-Calibrated LiRA t={timestep}")
        llr = gaussian_log_likelihood_ratio(
            train_df[columns].to_numpy(dtype=np.float64),
            y_train,
            eval_df[columns].to_numpy(dtype=np.float64),
        )
        col = f"llr_t{timestep}"
        out[col] = llr
        llr_columns.append(col)

    return out, llr_columns


def evaluate_timestep_calibrated_lira(
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    metrics = []
    rows = []

    for fold_id, seed in enumerate(SPLIT_SEEDS, start=1):
        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=seed,
            stratify=df["label"],
        )
        y_train = train_df["label"].to_numpy(dtype=int)
        y_test = test_df["label"].to_numpy(dtype=int)

        train_llr, llr_columns = idea2_features_for_split(train_df, train_df)
        test_llr, _ = idea2_features_for_split(train_df, test_df)

        clf = logistic_model(random_state=seed)
        clf.fit(train_llr[llr_columns], y_train)
        scores = clf.predict_proba(test_llr[llr_columns])[:, 1]
        pred = clf.predict(test_llr[llr_columns])

        test_llr["fold"] = fold_id
        test_llr["split_seed"] = seed
        test_llr["idea2_score"] = scores
        rows.append(test_llr)
        metrics.append(compute_metrics(y_test, scores, pred))

    return metrics, pd.concat(rows, ignore_index=True)


def evaluate_main_methods(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Series]]:
    require_columns(df, ["raw_loss"], "Raw Loss only")
    require_columns(df, tsva_v2_columns(), "TSVA-v2 only")
    require_columns(
        df,
        ["raw_loss", *HYBRID_TOP_TSVA_FEATURES],
        "Hybrid TSVA + Raw Loss",
    )

    rows = []
    feature_weights = []

    raw_metrics = evaluate_raw_score_group(df, "raw_loss", "Raw Loss only")
    rows.append(summarize_metrics("Raw Loss only", raw_metrics))

    tsva_metrics, tsva_weights = evaluate_logistic_group(df, tsva_v2_columns())
    rows.append(summarize_metrics("TSVA-v2 only", tsva_metrics))
    feature_weights.extend(tsva_weights)

    hybrid_columns = ["raw_loss", *HYBRID_TOP_TSVA_FEATURES]
    hybrid_metrics, hybrid_weights = evaluate_logistic_group(df, hybrid_columns)
    rows.append(summarize_metrics("Hybrid TSVA + Raw Loss", hybrid_metrics))
    feature_weights.extend(hybrid_weights)

    idea2_metrics, idea2_features = evaluate_timestep_calibrated_lira(df)
    rows.append(summarize_metrics("Timestep-Calibrated LiRA", idea2_metrics))

    return pd.DataFrame(rows)[RESULT_COLUMNS], idea2_features, feature_weights


def evaluate_named_groups(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []
    for name, columns in groups.items():
        metrics, _ = evaluate_logistic_group(df, columns)
        row = summarize_metrics(name, metrics)
        row["N_Features"] = len(columns)
        rows.append(row)
    return pd.DataFrame(rows)[["Method", "N_Features", *RESULT_COLUMNS[1:]]]


def temporal_ablation_groups() -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for timestep in TIMESTEPS:
        groups[f"timestep_{timestep}_raw_tsva"] = timestep_columns(timestep)
        groups[f"timestep_{timestep}_tsva_only"] = tsva_columns_for_timesteps([timestep])

    groups["early_raw_tsva"] = (
        raw_columns_for_timesteps(EARLY_TIMESTEPS)
        + tsva_columns_for_timesteps(EARLY_TIMESTEPS)
    )
    groups["mid_raw_tsva"] = (
        raw_columns_for_timesteps(MID_TIMESTEPS)
        + tsva_columns_for_timesteps(MID_TIMESTEPS)
    )
    groups["late_raw_tsva"] = (
        raw_columns_for_timesteps(LATE_TIMESTEPS)
        + tsva_columns_for_timesteps(LATE_TIMESTEPS)
    )
    groups["all_raw_tsva"] = raw_loss_columns() + tsva_v2_columns()
    return groups


def frequency_ablation_groups() -> Dict[str, List[str]]:
    return {
        "low_band_only": tsva_columns_for_bands(["low"]),
        "mid_band_only": tsva_columns_for_bands(["mid"]),
        "high_band_only": tsva_columns_for_bands(["high"]),
        "mid_high_bands": tsva_columns_for_bands(["mid", "high"]),
        "all_bands": tsva_v2_columns(),
    }


def statistic_ablation_groups() -> Dict[str, List[str]]:
    return {
        "mean_only": tsva_columns_for_stats(["mean"]),
        "std_only": tsva_columns_for_stats(["std"]),
        "mean_std": tsva_v2_columns(),
    }


def feature_importance_table(weights: List[pd.Series]) -> pd.DataFrame:
    combined = defaultdict(list)
    for series in weights:
        for feature, value in series.items():
            combined[feature].append(abs(float(value)))

    rows = []
    for feature, values in combined.items():
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "Feature": feature,
                "MeanAbsWeight": arr.mean(),
                "StdAbsWeight": arr.std(ddof=1) if len(arr) > 1 else 0.0,
                "SeenInSplits": len(arr),
            }
        )
    return pd.DataFrame(rows).sort_values("MeanAbsWeight", ascending=False)


def score_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw_score = -df["raw_loss"].to_numpy(dtype=np.float64)

    for feature in HYBRID_TOP_TSVA_FEATURES:
        values = df[feature].to_numpy(dtype=np.float64)
        corr = np.corrcoef(raw_score, values)[0, 1]
        rows.append({"ScoreA": "-raw_loss", "ScoreB": feature, "PearsonR": corr})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="std_results_labeled.csv")
    parser.add_argument("--results-out", default="research_results.csv")
    parser.add_argument("--idea2-out", default="idea2_features.csv")
    parser.add_argument("--temporal-out", default="temporal_ablation_results.csv")
    parser.add_argument("--frequency-out", default="frequency_ablation_results.csv")
    parser.add_argument("--statistic-out", default="statistic_ablation_results.csv")
    parser.add_argument("--importance-out", default="feature_importance_results.csv")
    parser.add_argument("--correlation-out", default="score_correlation_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected 'image' and 'label' columns in the input CSV.")
    df = prepare_balanced_data(df)

    main_results, idea2_features, weights = evaluate_main_methods(df)
    temporal_results = evaluate_named_groups(df, temporal_ablation_groups())
    frequency_results = evaluate_named_groups(df, frequency_ablation_groups())
    statistic_results = evaluate_named_groups(df, statistic_ablation_groups())
    importance_results = feature_importance_table(weights)
    correlation_results = score_correlation_table(df)

    main_results.to_csv(args.results_out, index=False)
    idea2_features.to_csv(args.idea2_out, index=False)
    temporal_results.to_csv(args.temporal_out, index=False)
    frequency_results.to_csv(args.frequency_out, index=False)
    statistic_results.to_csv(args.statistic_out, index=False)
    importance_results.to_csv(args.importance_out, index=False)
    correlation_results.to_csv(args.correlation_out, index=False)

    print("\n==============================")
    print("Research Method Comparison")
    print("==============================")
    print(main_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    for path in [
        args.results_out,
        args.idea2_out,
        args.temporal_out,
        args.frequency_out,
        args.statistic_out,
        args.importance_out,
        args.correlation_out,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
