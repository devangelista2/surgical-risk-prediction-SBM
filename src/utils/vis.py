from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

# Configure matplotlib for scientific publication quality
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
        "figure.dpi": 300,  # High resolution raster export
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)
sns.set_theme(style="whitegrid", context="paper")


def _save_figure(save_path: Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    # Always export a vector version for publication.
    plt.savefig(save_path.with_suffix(".pdf"))


def plot_confusion_matrix(y_true, y_pred, classes, title: str, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title, pad=15)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    sns.despine(offset=8)
    _save_figure(save_path)
    plt.close()


def plot_roc_curve(y_true_bin, y_prob_pos, title: str, save_path: Path):
    fpr, tpr, _ = roc_curve(y_true_bin, y_prob_pos)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="#d62728", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title, pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    sns.despine(offset=8)
    _save_figure(save_path)
    plt.close()


def plot_pr_curve(y_true_bin, y_prob_pos, title: str, save_path: Path):
    precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_pos)

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, color="#1f77b4", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title, pad=15)
    plt.tight_layout()
    sns.despine(offset=8)
    _save_figure(save_path)
    plt.close()


def plot_regression_scatter(y_true, y_pred, title: str, save_path: Path):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, color="#1f77b4", edgecolor="w", s=50)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title, pad=15)
    plt.legend(loc="upper left")
    plt.tight_layout()
    sns.despine(offset=8)
    _save_figure(save_path)
    plt.close()


def plot_feature_importance(
    features, importances, stds, title: str, save_path: Path, top_n: int = 20
):
    features = list(features)
    importances = np.asarray(importances, dtype=float)
    stds = np.asarray(stds, dtype=float)

    if len(features) == 0:
        return

    order = np.argsort(importances)[::-1]
    top_idx = order[:top_n]
    plot_features = [features[i] for i in top_idx][::-1]
    plot_importances = importances[top_idx][::-1]
    plot_stds = stds[top_idx][::-1]

    plt.figure(figsize=(8, max(4, len(plot_features) * 0.3)))
    plt.barh(
        plot_features,
        plot_importances,
        xerr=plot_stds,
        color="#1f77b4",
        alpha=0.85,
        ecolor="#4d4d4d",
    )
    plt.xlabel("Permutation Importance")
    plt.title(title, pad=15)
    plt.tight_layout()
    sns.despine(offset=8)
    _save_figure(save_path)
    plt.close()


# --- NEW COMBINED PLOTTING FUNCTIONS ---


def plot_combined_roc_curve(y_true_bin, y_prob_dict: dict, title: str, save_path: Path):
    plt.figure(figsize=(6, 6))

    # Loop through each model and plot its curve
    for model_name, y_prob_pos in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob_pos)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name.upper()} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title, pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    sns.despine(offset=8)
    _save_figure(save_path)
    plt.close()


def plot_combined_pr_curve(y_true_bin, y_prob_dict: dict, title: str, save_path: Path):
    plt.figure(figsize=(6, 6))

    # Loop through each model and plot its curve
    for model_name, y_prob_pos in y_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_pos)
        # Calculate Average Precision (AP) for the legend
        from sklearn.metrics import average_precision_score

        ap = average_precision_score(y_true_bin, y_prob_pos)
        plt.plot(recall, precision, lw=2, label=f"{model_name.upper()} (AP = {ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title, pad=15)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    sns.despine(offset=8)
    _save_figure(save_path)
    plt.close()
