from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

# Configure matplotlib for scientific publication quality
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,  # High resolution for papers
        "savefig.bbox": "tight",  # Prevent cropping of labels
    }
)
sns.set_theme(style="ticks", context="paper")


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
    sns.despine()
    plt.savefig(save_path)
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
    sns.despine()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(y_true_bin, y_prob_pos, title: str, save_path: Path):
    precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_pos)

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, color="#1f77b4", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title, pad=15)
    sns.despine()
    plt.savefig(save_path)
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
    sns.despine()
    plt.savefig(save_path)
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
    sns.despine()
    plt.savefig(save_path)
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
    sns.despine()
    plt.savefig(save_path)
    plt.close()
