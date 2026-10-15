import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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