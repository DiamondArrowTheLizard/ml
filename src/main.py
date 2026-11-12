import sys

from sklearn.model_selection import train_test_split as tts

import calc as cc
import classification_models as cm
import csv_to_dataframe as ctd
import dataset_description as dd
import describe_whole as dc
import describes as ds
import hyperparameter_tuning as ht
import model_comparison as mc
import pca_analysis as pa
import plots as pl

TARGET = "Wine"
PCA_VARIANCE_THRESHOLD = 0.95
PCA_MAX_COMPONENTS = 10
COMPARISON_CSV = "comp/model_comparison.csv"
TUNING_CSV = "comp/tuning_summary.csv"


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
        X, y, test_ratio=0.2, cv_folds=5, label="Raw features (default params)"
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
    print("=" * 70)
    print("HYPERPARAMETER TUNING PHASE")
    print("=" * 70)
    X_tr, X_te, y_tr, _y_te = tts(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"Tuning on training subset: {len(X_tr)} samples "
        f"(test subset {len(X_te)} untouched)"
    )
    tuning_report = ht.tune_all_models(
        X_tr, y_tr, cv_folds=5, random_state=42
    )
    print()
    print(tuning_report)

    print()
    tuned_overrides = tuning_report.as_estimators(random_state=42)
    tuned_report = cm.build_classification_models(
        X, y, test_ratio=0.2, cv_folds=5,
        estimator_overrides=tuned_overrides,
        label="Tuned (raw features)",
    )
    print(tuned_report)

    print()
    tuned_cm_paths = pl.plot_confusion_matrices_all(
        tuned_report.models, suffix="tuned_raw"
    )
    print("Saved confusion matrices (tuned, raw):")
    for p in tuned_cm_paths:
        print(f"  {p}")

    tuned_roc_paths = pl.plot_roc_curves_all(
        tuned_report.models, suffix="tuned_raw"
    )
    print("Saved ROC curves (tuned, raw):")
    for p in tuned_roc_paths:
        print(f"  {p}")

    print()
    sensitivity_paths = pl.plot_param_sensitivity_all(tuning_report)
    print("Saved parameter sensitivity plots:")
    for p in sensitivity_paths:
        print(f"  {p}")

    cmp_path = pl.plot_default_vs_tuned(
        raw_report, tuned_report, suffix="raw"
    )
    print(f"Saved default-vs-tuned bar chart: {cmp_path}")

    tuning_csv = ht.save_tuning_csv(tuning_report, TUNING_CSV)
    print(f"Saved tuning summary: {tuning_csv}")

    print()
    print(ht.compare_default_vs_tuned(raw_report, tuned_report))

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