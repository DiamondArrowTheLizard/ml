import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

DEFAULT_OUTPUT_DIR = "plots"


def _ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def plot_numeric_distributions(
    dataframe, numeric_columns, output_dir=DEFAULT_OUTPUT_DIR
):
    _ensure_output_dir(output_dir)
    saved = []
    for col in numeric_columns:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(dataframe[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        path = os.path.join(output_dir, f"dist_{col.replace(' ', '_')}.png")
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_categorical_distributions(
    dataframe, categorical_columns, output_dir=DEFAULT_OUTPUT_DIR, top_n=15
):
    _ensure_output_dir(output_dir)
    saved = []
    for col in categorical_columns:
        plt.close("all")
        counts = dataframe[col].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(counts))))
        sns.barplot(x=counts.values, y=counts.index, ax=ax)
        ax.set_title(f"Distribution of {col} (top {top_n})")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        fig.tight_layout()
        path = os.path.join(output_dir, f"dist_{col.replace(' ', '_')}.png")
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_correlation_matrix(correlation_matrix, output_dir=DEFAULT_OUTPUT_DIR):
    _ensure_output_dir(output_dir)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, square=True, ax=ax, annot_kws={"size": 7},
    )
    ax.set_title("Correlation Matrix (numeric features)")
    fig.tight_layout()
    path = os.path.join(output_dir, "correlation_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(",", "")
    )


def plot_confusion_matrix(model, output_dir=DEFAULT_OUTPUT_DIR, suffix: str = ""):
    _ensure_output_dir(output_dir)
    if model.confusion is None:
        return None
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        model.confusion, annot=True, fmt="d", cmap="Blues",
        xticklabels=model.classes, yticklabels=model.classes,
        square=True, cbar=False, ax=ax,
    )
    title = f"Confusion Matrix — {model.name}"
    if suffix:
        title += f" ({suffix})"
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    safe = _safe_name(model.name)
    if suffix:
        safe = f"{safe}_{suffix}"
    path = os.path.join(output_dir, f"confusion_{safe}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrices_all(models, output_dir=DEFAULT_OUTPUT_DIR, suffix: str = ""):
    _ensure_output_dir(output_dir)
    saved = []
    for model in models:
        path = plot_confusion_matrix(model, output_dir, suffix=suffix)
        if path is not None:
            saved.append(path)
    return saved


def plot_roc_curves(model, output_dir=DEFAULT_OUTPUT_DIR, suffix: str = ""):
    _ensure_output_dir(output_dir)
    if model.y_test is None or model.y_proba is None or model.proba_classes is None:
        return None

    plt.close("all")
    classes = list(model.proba_classes)
    y_bin = label_binarize(model.y_test, classes=classes)
    if y_bin.ndim == 1 or y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin.reshape(-1, 1), y_bin.reshape(-1, 1)])

    fig, ax = plt.subplots(figsize=(7, 6))

    fpr_all = {}
    tpr_all = {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], model.y_proba[:, i])
        fpr_all[cls] = fpr
        tpr_all[cls] = tpr
        cls_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"Class {cls} (AUC = {cls_auc:.3f})")

    all_fpr = np.unique(np.concatenate([fpr_all[c] for c in classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls in classes:
        mean_tpr += np.interp(all_fpr, fpr_all[cls], tpr_all[cls])
    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr, mean_tpr, color="black", linestyle="--", linewidth=2.2,
        label=f"Macro-avg (AUC = {macro_auc:.3f})",
    )

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    title = f"ROC Curves — {model.name}"
    if suffix:
        title += f" ({suffix})"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    safe = _safe_name(model.name)
    if suffix:
        safe = f"{safe}_{suffix}"
    path = os.path.join(output_dir, f"roc_{safe}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_roc_curves_all(models, output_dir=DEFAULT_OUTPUT_DIR, suffix: str = ""):
    _ensure_output_dir(output_dir)
    saved = []
    for model in models:
        path = plot_roc_curves(model, output_dir, suffix=suffix)
        if path is not None:
            saved.append(path)
    return saved


def plot_metric_comparison(report, output_dir=DEFAULT_OUTPUT_DIR, suffix: str = ""):
    _ensure_output_dir(output_dir)
    plt.close("all")

    names = [m.name for m in report.models]
    metrics = {
        "Accuracy": [m.accuracy for m in report.models],
        "Precision (macro)": [m.precision_macro for m in report.models],
        "Recall (macro)": [m.recall_macro for m in report.models],
        "F1 (macro)": [m.f1_macro for m in report.models],
        "AUC-ROC (OvR)": [
            m.roc_auc_ovr_macro if m.roc_auc_ovr_macro is not None else np.nan
            for m in report.models
        ],
    }
    x = np.arange(len(names))
    n_metrics = len(metrics)
    width = 0.16
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (label, values) in enumerate(metrics.items()):
        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    title = "Model comparison — test metrics"
    if suffix:
        title += f" ({suffix})"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    safe = f"metric_comparison_{suffix}" if suffix else "metric_comparison"
    path = os.path.join(output_dir, f"{safe}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_scree(pca_result, output_dir=DEFAULT_OUTPUT_DIR, max_display: int = 10):
    _ensure_output_dir(output_dir)
    plt.close("all")

    ratios = pca_result.explained_variance_ratio[:max_display]
    cumulative = pca_result.cumulative_variance_ratio[:max_display]
    labels = pca_result.component_names()[:max_display]
    x = np.arange(len(ratios))

    fig, ax1 = plt.subplots(figsize=(11, 5))
    bars = ax1.bar(
        x, ratios, width=0.55, alpha=0.85, color="steelblue",
        edgecolor="black", linewidth=0.6, label="Individual variance",
    )
    for rect, value in zip(bars, ratios):
        ax1.text(
            rect.get_x() + rect.get_width() / 2, value, f"{value:.1%}",
            ha="center", va="bottom", fontsize=8,
        )

    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio (individual)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(ratios) * 1.2 if len(ratios) else 1.0)

    ax2 = ax1.twinx()
    ax2.plot(
        x, cumulative, color="darkred", marker="o", markersize=6,
        linewidth=1.8, label="Cumulative variance",
    )
    for xi, value in zip(x, cumulative):
        ax2.annotate(
            f"{value:.1%}", (xi, value), textcoords="offset points",
            xytext=(0, 8), ha="center", fontsize=8, color="darkred",
        )
    ax2.axhline(
        y=pca_result.variance_threshold, color="green", linestyle="--",
        linewidth=1.2, label=f"Threshold = {pca_result.variance_threshold:.2f}",
    )
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    total_kept = pca_result.n_components
    total_var = pca_result.cumulative_variance_ratio[-1]
    ax1.set_title(
        f"Scree Plot — showing first {len(ratios)} of {total_kept} components "
        f"(total variance explained by kept components: {total_var:.2%})"
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "pca_scree.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cumulative_variance(pca_result, output_dir=DEFAULT_OUTPUT_DIR):
    _ensure_output_dir(output_dir)
    plt.close("all")
    cumulative = pca_result.cumulative_variance_ratio
    x = np.arange(len(cumulative))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, cumulative, color="darkred", marker="o", markersize=4, linewidth=1.5)
    ax.axhline(
        y=pca_result.variance_threshold, color="green", linestyle="--",
        linewidth=1.2, label=f"Threshold = {pca_result.variance_threshold:.2f}",
    )
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    ax.set_title("PCA — cumulative explained variance")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "pca_cumulative_variance.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_plots(description, output_dir=DEFAULT_OUTPUT_DIR):
    numeric = plot_numeric_distributions(
        description.dataframe, description.numeric_columns, output_dir
    )
    categorical = plot_categorical_distributions(
        description.dataframe, description.categorical_columns, output_dir
    )
    correlation = plot_correlation_matrix(description.correlation_matrix, output_dir)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "correlation": [correlation],
    }