# main.py
import sys

import calc as cc
import crosstabs as ct
import csv_to_dataframe as ctd
import dataset_description as dd
import describe_whole as dc
import describes as ds
import plots as pl
import regression_models as rm

CATEGORICAL_VARS = ["Gender", "Education Level", "Job Title"]
TARGET = "Salary"


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

    print()
    X, y = rm.prepare_features(data, TARGET)
    report = rm.build_regression_models(X, y, test_ratio=0.2, cv_folds=5)
    print(report)


if __name__ == "__main__":
    main(sys.argv)