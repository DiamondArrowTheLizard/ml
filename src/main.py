import sys

import calc as cc
import classification_models as cm
import csv_to_dataframe as ctd
import dataset_description as dd
import describe_whole as dc
import describes as ds
import model_comparison as mc
import pca_analysis as pa
import plots as pl

TARGET = "Wine"
PCA_VARIANCE_THRESHOLD = 0.95
PCA_MAX_COMPONENTS = 10
COMPARISON_CSV = "comp/model_comparison.csv"


def main(argv):
    if len(argv) < 2:
        print("No .csv specified.")
        return

    data = ctd.csv_to_dataframe(argv[1])
    data = data.dropna(how="all").reset_index(drop=True)
    data = data.dropna().reset_index(drop=True)

    print(data)
    print()
    dc.describe_whole_print_dataset(data)
    print()
    print(f"Sturges rule bins: {cc.calc_sturges_rule(len(data))}")
    print()
    for d in ds.generate_describes(data):
        print(d)
        print()

    description = dd.DatasetDescription(data)
    print(description)
    print()

    saved = pl.generate_all_plots(description)
    print("Saved plots:")
    for group, paths in saved.items():
        for path in paths:
            print(f"  [{group}] {path}")

    X, y = cm.prepare_features(data, TARGET)
    print()
    print(
        f"Features: {X.shape[1]}, "
        f"Classes: {[int(c) for c in sorted(y.unique())]}, Samples: {len(y)}"
    )

    print()
    raw_report = cm.build_classification_models(
        X, y, test_ratio=0.2, cv_folds=5, label="Raw features"
    )
    print(raw_report)

    print()
    cm_paths = pl.plot_confusion_matrices_all(raw_report.models, suffix="raw")
    print("Saved confusion matrices (raw):")
    for p in cm_paths:
        print(f"  {p}")

    roc_paths = pl.plot_roc_curves_all(raw_report.models, suffix="raw")
    print("Saved ROC curves (raw):")
    for p in roc_paths:
        print(f"  {p}")

    metric_path = pl.plot_metric_comparison(raw_report, suffix="raw")
    print(f"Saved metric comparison (raw): {metric_path}")

    print()
    pca_result = pa.run_pca_analysis(
        X, y,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
        max_components=PCA_MAX_COMPONENTS,
    )
    print(pca_result)

    scree_path = pl.plot_scree(
        pca_result, max_display=min(PCA_MAX_COMPONENTS, pca_result.n_components)
    )
    cumulative_path = pl.plot_cumulative_variance(pca_result)
    print()
    print(f"Saved scree plot: {scree_path}")
    print(f"Saved cumulative variance plot: {cumulative_path}")

    print()
    pca_report = cm.build_classification_models(
        pca_result.components_df, y,
        test_ratio=0.2, cv_folds=5,
        label=f"PCA ({pca_result.n_components} components, "
              f"{pca_result.variance_threshold:.0%} variance)",
    )
    print(pca_report)

    print()
    pca_cm_paths = pl.plot_confusion_matrices_all(pca_report.models, suffix="pca")
    print("Saved confusion matrices (PCA):")
    for p in pca_cm_paths:
        print(f"  {p}")

    pca_roc_paths = pl.plot_roc_curves_all(pca_report.models, suffix="pca")
    print("Saved ROC curves (PCA):")
    for p in pca_roc_paths:
        print(f"  {p}")

    pca_metric_path = pl.plot_metric_comparison(pca_report, suffix="pca")
    print(f"Saved metric comparison (PCA): {pca_metric_path}")

    print()
    comparison = mc.ModelComparison(raw_report, pca_report)
    print(comparison)

    csv_path = mc.save_comparison_csv(comparison, COMPARISON_CSV)
    print()
    print(f"Saved comparison table: {csv_path}")

    print()
    try:
        import nn_classifier as nn
        print("Running TensorFlow neural network demo...")
        nn_result = nn.build_nn_classifier(data, TARGET, epochs=100)
        print(nn_result)
    except ImportError:
        print("TensorFlow not installed — skipping NN part.")
        print("Install with: pip install tensorflow tensorboard")


if __name__ == "__main__":
    main(sys.argv)