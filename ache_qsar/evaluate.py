"""Metrics and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, mean_absolute_error, precision_score, r2_score, recall_score,
    roc_auc_score, root_mean_squared_error,
)


def clf_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
    }


def reg_metrics(y_true, pred) -> dict:
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    constant = np.ptp(pred) < 1e-9
    return {
        "r2": r2_score(y_true, pred),
        "rmse": root_mean_squared_error(y_true, pred),
        "mae": mean_absolute_error(y_true, pred),
        "pearson_r": np.nan if constant else pearsonr(y_true, pred)[0],
        "spearman_rho": np.nan if constant else spearmanr(y_true, pred)[0],
    }


def bootstrap_ci(y_true, y_hat, metric_fn, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI over test compounds for every metric returned by `metric_fn`."""
    y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        if len(np.unique(yt)) < 2 and set(np.unique(y_true)) == {0, 1}:
            continue   # classification resample with a single class: AUC undefined
        samples.append(metric_fn(yt, y_hat[idx]))
    keys = samples[0].keys()
    return {k: (float(np.nanpercentile([s[k] for s in samples], 100 * alpha / 2)),
                float(np.nanpercentile([s[k] for s in samples], 100 * (1 - alpha / 2)))) for k in keys}
