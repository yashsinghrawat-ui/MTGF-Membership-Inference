import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TIMESTEPS = [50, 100, 150, 200, 300, 400, 500, 750]
EARLY_TIMESTEPS = [50, 100, 150, 200]
LATE_TIMESTEPS = [400, 500, 750]
BANDS = ("low", "mid", "high")


def _present(df, columns):
    return [c for c in columns if c in df.columns]


def _cols(prefix, timesteps=TIMESTEPS):
    return [f"{prefix}_{t}" for t in timesteps]


def add_curve_features(df):
    out = df.copy()

    raw_cols = _present(out, _cols("raw_loss_mean"))
    if raw_cols:
        early_raw = _present(out, _cols("raw_loss_mean", EARLY_TIMESTEPS))
        late_raw = _present(out, _cols("raw_loss_mean", LATE_TIMESTEPS))

        out["raw_curve_auc"] = out[raw_cols].mean(axis=1)

        if early_raw and late_raw:
            out["raw_late_minus_early"] = (
                out[late_raw].mean(axis=1) - out[early_raw].mean(axis=1)
            )
            out["raw_late_over_early"] = (
                out[late_raw].mean(axis=1) / (out[early_raw].mean(axis=1) + 1e-8)
            )

        x = np.asarray([int(c.rsplit("_", 1)[1]) for c in raw_cols], dtype=np.float64)
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = out[raw_cols].to_numpy(dtype=np.float64)
        out["raw_curve_slope"] = y @ x / (x @ x + 1e-8)

    for band in BANDS:
        mean_cols = _present(out, _cols(f"{band}_mean"))
        if not mean_cols:
            continue

        early_cols = _present(out, _cols(f"{band}_mean", EARLY_TIMESTEPS))
        late_cols = _present(out, _cols(f"{band}_mean", LATE_TIMESTEPS))

        out[f"{band}_curve_auc"] = out[mean_cols].mean(axis=1)

        if early_cols and late_cols:
            out[f"{band}_late_minus_early"] = (
                out[late_cols].mean(axis=1) - out[early_cols].mean(axis=1)
            )
            out[f"{band}_late_over_early"] = (
                out[late_cols].mean(axis=1) / (out[early_cols].mean(axis=1) + 1e-8)
            )

        x = np.asarray([int(c.rsplit("_", 1)[1]) for c in mean_cols], dtype=np.float64)
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = out[mean_cols].to_numpy(dtype=np.float64)
        out[f"{band}_curve_slope"] = y @ x / (x @ x + 1e-8)

    if all(f"{band}_curve_auc" in out.columns for band in BANDS):
        out["high_mid_curve_ratio"] = (
            out["high_curve_auc"] / (out["mid_curve_auc"] + 1e-8)
        )
        out["mid_low_curve_ratio"] = (
            out["mid_curve_auc"] / (out["low_curve_auc"] + 1e-8)
        )

    return out


def feature_groups(df):
    raw = [c for c in df.columns if c.startswith("raw_loss_")]
    tsva = [
        c for c in df.columns
        if re.match(r"^(low|mid|high)_(mean|std)_\d+$", c)
    ]
    late = [
        c for c in raw + tsva
        if int(c.rsplit("_", 1)[1]) in LATE_TIMESTEPS
    ]
    curve = [
        c for c in df.columns
        if c.endswith("_auc")
        or c.endswith("_slope")
        or c.endswith("_late_minus_early")
        or c.endswith("_late_over_early")
        or c.endswith("_curve_ratio")
    ]

    groups = {
        "raw_loss_curve": raw,
        "tsva_spectral_curve": tsva,
        "late_tsva_raw": late,
        "curve_summary": curve,
        "all_profile_features": sorted(set(raw + tsva + curve)),
    }

    return {k: v for k, v in groups.items() if v}


def evaluate_group(X, y, columns):
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    aucs = []
    accs = []

    for train_idx, test_idx in cv.split(X[columns], y):
        X_train = X.iloc[train_idx][columns]
        X_test = X.iloc[test_idx][columns]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=10000,
                class_weight="balanced"
            )
        )

        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_test, prob))
        accs.append(accuracy_score(y_test, pred))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
    }


def fit_holdout(X, y, columns):
    X_train, X_test, y_train, y_test = train_test_split(
        X[columns],
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=10000,
            class_weight="balanced"
        )
    )
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    weights = pd.Series(
        clf.named_steps["logisticregression"].coef_[0],
        index=columns
    ).sort_values(key=np.abs, ascending=False)

    return roc_auc_score(y_test, prob), accuracy_score(y_test, pred), weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="std_results_labeled.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("Expected a labeled CSV with a 'label' column.")

    df = add_curve_features(df)
    groups = feature_groups(df)

    y = df["label"].astype(int)
    X = df.drop(columns=[c for c in ("image", "label") if c in df.columns])

    print("\n==============================")
    print("LEAKAGE PROFILE ABLATIONS")
    print("==============================")

    rows = []
    for name, columns in groups.items():
        metrics = evaluate_group(X, y, columns)
        rows.append({"group": name, "n_features": len(columns), **metrics})

    result = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_name = result.iloc[0]["group"]
    best_cols = groups[best_name]
    auc, acc, weights = fit_holdout(X, y, best_cols)

    print("\n==============================")
    print(f"BEST HOLDOUT MODEL: {best_name}")
    print("==============================")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC     : {auc:.4f}")

    print("\nTop features:")
    print(weights.head(20).to_string())


if __name__ == "__main__":
    main()
