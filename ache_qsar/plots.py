"""Figures. Each function saves a PNG and returns the path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit.Chem import Descriptors
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from . import config
from .features import to_mols

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"active": "#2a9d8f", "inactive": "#e76f51", "intermediate": "#b0b0b0"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_eda(df: pd.DataFrame, path: Path, target_label: str = "AChE") -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].hist(df["pIC50"], bins=40, color="#264653", edgecolor="white")
    ax[0].axvline(config.ACTIVE_PIC50, color=PALETTE["active"], ls="--", label=f"active (pIC50 ≥ {config.ACTIVE_PIC50:g})")
    ax[0].axvline(config.INACTIVE_PIC50, color=PALETTE["inactive"], ls="--", label=f"inactive (pIC50 ≤ {config.INACTIVE_PIC50:g})")
    ax[0].set(title=f"{target_label}: pIC50 distribution (n = {len(df)})", xlabel="pIC50", ylabel="Compounds")
    ax[0].legend(fontsize=8)

    order = ["active", "intermediate", "inactive"]
    counts = df["activity_class"].value_counts().reindex(order).fillna(0)
    ax[1].bar(order, counts.values, color=[PALETTE[o] for o in order], edgecolor="black", linewidth=0.6)
    for i, v in enumerate(counts.values):
        ax[1].text(i, v, int(v), ha="center", va="bottom", fontweight="bold")
    ax[1].set(title="Activity classes", ylabel="Compounds")

    mols = to_mols(df["smiles"])
    phys = pd.DataFrame({"MW": [Descriptors.MolWt(m) for m in mols],
                         "cLogP": [Descriptors.MolLogP(m) for m in mols],
                         "class": df["activity_class"].values})
    sns.scatterplot(data=phys, x="MW", y="cLogP", hue="class", palette=PALETTE, alpha=0.5, s=18, ax=ax[2], edgecolor=None)
    ax[2].set(title="Chemical space (MW vs cLogP)")
    return _save(fig, path)


def plot_model_comparison(df: pd.DataFrame, task: str, path: Path) -> Path:
    cv_col, cv_sd, test_col, label = (
        ("cv_roc_auc", "cv_roc_auc_std", "test_roc_auc", "ROC-AUC") if task == "classification"
        else ("cv_rmse", "cv_rmse_std", "test_rmse", "RMSE (pIC50 units)"))
    d = df.copy()
    d["label"] = d["model"] + "\n(" + d["feature_set"] + ")"
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(d)), 5))
    ax.bar(x - 0.2, d[cv_col], 0.4, yerr=d[cv_sd], capsize=3, label="Scaffold-grouped CV (train)", color="#8ecae6", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, d[test_col], 0.4, label="Held-out scaffold test set", color="#023047", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], rotation=60, ha="right", fontsize=8)
    ax.set(ylabel=label, title=f"{task.capitalize()}: CV vs. test performance")
    if task == "classification":
        ax.axhline(0.5, color="grey", ls="--", lw=1, label="Random (AUC 0.5)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_roc(y_true, proba_by_model: dict, path: Path) -> Path:
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, p in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves - held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_confusion(y_true, proba, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ConfusionMatrixDisplay.from_predictions(y_true, (np.asarray(proba) >= 0.5).astype(int),
                                            display_labels=["Inactive", "Active"], cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    return _save(fig, path)


def plot_pred_vs_actual(y_true, pred, metrics: dict, path: Path, title: str) -> Path:
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    lo, hi = min(y_true.min(), pred.min()) - 0.3, max(y_true.max(), pred.max()) + 0.3
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.scatter(y_true, pred, s=22, alpha=0.55, color="#264653", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "r-", lw=1.4, label="Ideal")
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="red", alpha=0.08, label="±1 log unit")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Experimental pIC50", ylabel="Predicted pIC50", title=title)
    ax.text(0.04, 0.96, f"R² = {metrics['r2']:.2f}\nRMSE = {metrics['rmse']:.2f}\nMAE = {metrics['mae']:.2f}",
            transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="grey"))
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_y_randomization(real: float, scrambled, metric: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(scrambled, bins=12, color="#e76f51", edgecolor="white", label="Y-scrambled models")
    ax.axvline(real, color="#023047", lw=2.5, label=f"Real model ({real:.3f})")
    ax.set(xlabel=f"Test {metric}", ylabel="Count", title="Y-randomisation (chance-correlation check)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_applicability_domain(sim, abs_err, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.scatter(sim, abs_err, s=20, alpha=0.5, color="#264653", edgecolors="none")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="1 log unit")
    ax.set(xlabel="Max Tanimoto similarity to training set (ECFP4)", ylabel="|Prediction error| (pIC50)",
           title="Applicability domain: error vs. novelty")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_importance(importances: pd.Series, path: Path, title: str) -> Path:
    top = importances.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(top.index, top.values, color="#2a9d8f", edgecolor="black", linewidth=0.4)
    ax.set(xlabel="Permutation importance (increase in RMSE)", title=title)
    return _save(fig, path)
