import sys

import calc as cc
import crosstabs as ct
import csv_to_dataframe as ctd
import dataset_description as dd
import describe_whole as dc
import describes as ds
import heteroscedasticity as hs
import model_comparison as mc
import pca_analysis as pa
import plots as pl
import regression_models as rm

CATEGORICAL_VARS = ["Gender", "Education Level", "Job Title"]
TARGET = "Salary"
MAX_CATEGORIES = 10
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
    for describe in ds.generate_describes(data):
        print(describe)
        print()

    ct.print_crosstabs(ct.generate_crosstabs(data, CATEGORICAL_VARS))

    print()
    description = dd.DatasetDescription(data)
    print(description)

    print()
    saved = pl.generate_all_plots(description)
    print("Saved plots:")
    for group, paths in saved.items():
        for path in paths:
            print(f"  [{group}] {path}")

    X, y = rm.prepare_features(data, TARGET, max_categories=MAX_CATEGORIES)
    print()
    print(
        f"Features after one-hot encoding (max {MAX_CATEGORIES} categories/col): "
        f"{X.shape[1]}"
    )

    print()
    raw_report = rm.build_regression_models(
        X, y, test_ratio=0.2, cv_folds=5, label="Raw features"
    )
    print(raw_report)

    print()
    residual_paths = pl.plot_residuals_all(raw_report.models, suffix="raw")
    print("Saved residual plots (raw features):")
    for path in residual_paths:
        print(f"  {path}")

    print()
    het_report = hs.analyse_heteroscedasticity(raw_report.models)
    print(het_report)

    print()
    pca_result = pa.run_pca_analysis(
        X,
        y,
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
    pca_report = rm.build_regression_models(
        pca_result.components_df,
        y,
        test_ratio=0.2,
        cv_folds=5,
        label=f"PCA ({pca_result.n_components} components, "
        f"{pca_result.variance_threshold:.0%} variance)",
    )
    print(pca_report)

    print()
    pca_residual_paths = pl.plot_residuals_all(pca_report.models, suffix="pca")
    print("Saved residual plots (PCA features):")
    for path in pca_residual_paths:
        print(f"  {path}")

    print()
    pca_het_report = hs.analyse_heteroscedasticity(pca_report.models)
    print(pca_het_report)

    print()
    comparison = mc.ModelComparison(raw_report, pca_report)
    print(comparison)

    csv_path = mc.save_comparison_csv(comparison, COMPARISON_CSV)
    print()
    print(f"Saved comparison table: {csv_path}")


if __name__ == "__main__":
    main(sys.argv)
