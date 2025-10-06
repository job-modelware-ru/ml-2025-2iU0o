#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

# ---------------------------
# Config
# ---------------------------
RESULTS_DIR = "results"
RANDOM_STATE = 42
N_JOBS = 1
N_SPLITS = 5
POS_LABEL = 1

np.random.seed(RANDOM_STATE)


@dataclass
class SchemeResults:
    accuracy: float
    f1: float
    roc_auc: float


def ensure_results_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_pipeline() -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        bootstrap=True,
        oob_score=True,  # used only when fitting on the whole data for OOB
    )
    return Pipeline([("rf", rf)])


def positive_index(estimator: RandomForestClassifier, pos_label: int) -> int:
    classes = estimator.classes_
    idx = np.where(classes == pos_label)[0]
    if idx.size == 0:
        raise ValueError(f"POS_LABEL={pos_label} is not present in estimator.classes_={classes!r}")
    return int(idx[0])


def evaluate_holdout(X: np.ndarray, y: np.ndarray) -> Tuple[SchemeResults, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    est: RandomForestClassifier = pipe.named_steps["rf"]
    pos_idx = positive_index(est, POS_LABEL)

    y_prob = est.predict_proba(X_test)[:, pos_idx]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL)
    roc = roc_auc_score(y_test, y_prob)

    return SchemeResults(acc, f1, roc), {
        "holdout_accuracy": float(acc),
        "holdout_f1": float(f1),
        "holdout_roc_auc": float(roc),
    }


def evaluate_kfold(X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS) -> Tuple[pd.DataFrame, Dict[str, float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows: List[Dict[str, float]] = []
    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        est: RandomForestClassifier = pipe.named_steps["rf"]
        pos_idx = positive_index(est, POS_LABEL)

        y_prob = est.predict_proba(X_te)[:, pos_idx]
        y_pred = (y_prob >= 0.5).astype(int)

        rows.append(
            {
                "fold": fold_id,
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "f1": float(f1_score(y_te, y_pred, pos_label=POS_LABEL)),
                "roc_auc": float(roc_auc_score(y_te, y_prob)),
            }
        )

    df = pd.DataFrame(rows)

    means = df[["accuracy", "f1", "roc_auc"]].mean()
    stds = df[["accuracy", "f1", "roc_auc"]].std(ddof=1)

    summary = {
        "kfold_accuracy_mean": float(means["accuracy"]),
        "kfold_accuracy_std": float(stds["accuracy"]),
        "kfold_f1_mean": float(means["f1"]),
        "kfold_f1_std": float(stds["f1"]),
        "kfold_roc_auc_mean": float(means["roc_auc"]),
        "kfold_roc_auc_std": float(stds["roc_auc"]),
    }
    return df, summary


def evaluate_oob(X: np.ndarray, y: np.ndarray) -> Tuple[SchemeResults, Dict[str, float]]:
    pipe = build_pipeline()
    pipe.fit(X, y)

    est: RandomForestClassifier = pipe.named_steps["rf"]
    if not hasattr(est, "oob_decision_function_") or est.oob_decision_function_ is None:
        raise RuntimeError("oob_decision_function_ is not available; check bootstrap=True and oob_score=True.")

    pos_idx = positive_index(est, POS_LABEL)
    y_prob_oob = est.oob_decision_function_[:, pos_idx]
    y_pred_oob = (y_prob_oob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred_oob)  # computed from thresholded OOB probs
    f1 = f1_score(y, y_pred_oob, pos_label=POS_LABEL)
    roc = roc_auc_score(y, y_prob_oob)

    return SchemeResults(acc, f1, roc), {
        "oob_accuracy": float(acc),
        "oob_f1": float(f1),
        "oob_roc_auc": float(roc),
    }


def plot_metric_box(metric_name: str, kfold_values: List[float], holdout_value: float, oob_value: float, filepath: str) -> None:
    plt.figure()
    plt.boxplot(kfold_values, vert=True, labels=[metric_name])

    lo = min(float(np.min(kfold_values)), float(holdout_value), float(oob_value))
    hi = max(float(np.max(kfold_values)), float(holdout_value), float(oob_value))
    pad = max(1e-3, 0.02 * (hi - lo))

    plt.hlines(holdout_value, 0.75, 1.25, linestyles="--")
    plt.hlines(oob_value, 0.75, 1.25, linestyles=":")
    plt.title(f"{metric_name}: k-fold distribution; dashed = hold-out; dotted = OOB")
    plt.ylabel(metric_name)
    plt.ylim(lo - pad, hi + pad)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()


def main() -> None:
    ensure_results_dir(RESULTS_DIR)

    # ---------------------------
    # Data
    # ---------------------------
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Safety check (POS_LABEL)
    unique_labels = np.unique(y)
    if POS_LABEL not in unique_labels:
        # If POS_LABEL is not present, fall back to the largest label by convention
        raise ValueError(f"Configured POS_LABEL={POS_LABEL} not in dataset labels={unique_labels.tolist()}")

    # ---------------------------
    # Evaluate
    # ---------------------------
    holdout_res, holdout_summary = evaluate_holdout(X, y)
    kfold_df, kfold_summary = evaluate_kfold(X, y, n_splits=N_SPLITS)
    oob_res, oob_summary = evaluate_oob(X, y)

    # ---------------------------
    # Persist results
    # ---------------------------
    # Fold-wise CSV
    kfold_csv = os.path.join(RESULTS_DIR, "kfold_metrics.csv")
    kfold_df.to_csv(kfold_csv, index=False)

    # Hold-out CSV (single row)
    holdout_csv = os.path.join(RESULTS_DIR, "holdout_metrics.csv")
    pd.DataFrame([{
        "accuracy": holdout_res.accuracy,
        "f1": holdout_res.f1,
        "roc_auc": holdout_res.roc_auc,
    }]).to_csv(holdout_csv, index=False)

    # OOB CSV (single row)
    oob_csv = os.path.join(RESULTS_DIR, "oob_metrics.csv")
    pd.DataFrame([{
        "accuracy": oob_res.accuracy,
        "f1": oob_res.f1,
        "roc_auc": oob_res.roc_auc,
    }]).to_csv(oob_csv, index=False)

    # Summary CSV (flat, convenient for slides)
    summary_rows = [
        {"scheme": "hold-out", "metric": "accuracy", "value": holdout_res.accuracy},
        {"scheme": "hold-out", "metric": "f1", "value": holdout_res.f1},
        {"scheme": "hold-out", "metric": "roc_auc", "value": holdout_res.roc_auc},
        {"scheme": f"k-fold ({N_SPLITS})", "metric": "accuracy_mean", "value": kfold_summary["kfold_accuracy_mean"]},
        {"scheme": f"k-fold ({N_SPLITS})", "metric": "accuracy_std", "value": kfold_summary["kfold_accuracy_std"]},
        {"scheme": f"k-fold ({N_SPLITS})", "metric": "f1_mean", "value": kfold_summary["kfold_f1_mean"]},
        {"scheme": f"k-fold ({N_SPLITS})", "metric": "f1_std", "value": kfold_summary["kfold_f1_std"]},
        {"scheme": f"k-fold ({N_SPLITS})", "metric": "roc_auc_mean", "value": kfold_summary["kfold_roc_auc_mean"]},
        {"scheme": f"k-fold ({N_SPLITS})", "metric": "roc_auc_std", "value": kfold_summary["kfold_roc_auc_std"]},
        {"scheme": "OOB", "metric": "accuracy", "value": oob_res.accuracy},
        {"scheme": "OOB", "metric": "f1", "value": oob_res.f1},
        {"scheme": "OOB", "metric": "roc_auc", "value": oob_res.roc_auc},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(RESULTS_DIR, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # JSON report (includes small note for OOB)
    report = {
        "config": {
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
            "n_splits": N_SPLITS,
            "pos_label": POS_LABEL,
            "dataset": "sklearn.datasets.load_breast_cancer",
            "model": "RandomForestClassifier(bootstrap=True, oob_score=True)",
        },
        "holdout": holdout_summary,
        "kfold": kfold_summary,
        "oob": oob_summary,
        "notes": {
            "oob": "OOB metrics use oob_decision_function_ probabilities; threshold=0.5 for accuracy/F1 to keep parity with hold-out and k-fold.",
            "reproducibility": "For bitwise-stable results set n_jobs=1; parallel RNG may cause tiny jitter otherwise.",
            "pipeline": "All fit-dependent transforms must live inside a Pipeline/ColumnTransformer to avoid leakage.",
        },
    }
    report_json = os.path.join(RESULTS_DIR, "report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---------------------------
    # Plots
    # ---------------------------
    # Boxplots with hold-out (dashed) + OOB (dotted) overlays
    acc_png = os.path.join(RESULTS_DIR, "box_acc.png")
    f1_png = os.path.join(RESULTS_DIR, "box_f1.png")
    roc_png = os.path.join(RESULTS_DIR, "box_roc_auc.png")

    plot_metric_box("Accuracy", kfold_df["accuracy"].tolist(), holdout_res.accuracy, oob_res.accuracy, acc_png)
    plot_metric_box("F1", kfold_df["f1"].tolist(), holdout_res.f1, oob_res.f1, f1_png)
    plot_metric_box("ROC-AUC", kfold_df["roc_auc"].tolist(), holdout_res.roc_auc, oob_res.roc_auc, roc_png)

    # Small stdout summary
    print("=== Hold-out ===")
    print(holdout_summary)
    print("\n=== k-fold (means ± std) ===")
    for k, v in kfold_summary.items():
        print(f"{k}: {v:.6f}")
    print("\n=== OOB ===")
    print(oob_summary)
    print(f"\nArtifacts saved to: {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
