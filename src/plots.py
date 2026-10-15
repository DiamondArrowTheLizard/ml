import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_OUTPUT_DIR = "plots"


def _ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def plot_numeric_distributions(dataframe, numeric_columns, output_dir=DEFAULT_OUTPUT_DIR):
    _ensure_output_dir(output_dir)
    saved = []
    for col in numeric_columns:
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
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Matrix (numeric features)")
    fig.tight_layout()
    path = os.path.join(output_dir, "correlation_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_residuals(model, output_dir=DEFAULT_OUTPUT_DIR):
    """Residuals vs fitted values with a reference line at 0."""
    _ensure_output_dir(output_dir)
    if model.residuals is None or model.y_pred is None:
        return None

    fitted = np.asarray(model.y_pred).ravel()
    residuals = np.asarray(model.residuals).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, residuals, alpha=0.5, edgecolor="k", linewidth=0.3)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="y = 0")
    ax.set_title(f"Residuals vs Fitted — {model.name}")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.legend()
    fig.tight_layout()

    safe_name = model.name.replace(" ", "_")
    path = os.path.join(output_dir, f"residuals_{safe_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_residuals_all(models, output_dir=DEFAULT_OUTPUT_DIR):
    _ensure_output_dir(output_dir)
    saved = []
    for model in models:
        path = plot_residuals(model, output_dir)
        if path is not None:
            saved.append(path)
    return saved


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