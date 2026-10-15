import sys

import crosstabs as ct
import csv_to_dataframe as ctd
import describe_whole as dc
import describes as ds

CATEGORICAL_VARS = ["Gender", "Education Level", "Job Title"]


def main(argv):
    if len(argv) < 2:
        print("No .csv specified.")
        return

    data = ctd.csv_to_dataframe(argv[1])
    data = data.dropna(how="all").reset_index(drop=True)
    data = data.dropna().reset_index(drop=True)

    print(data)

    dc.describe_whole_print_dataset(data)

    for describe in ds.generate_describes(data):
        print(describe)

    ct.print_crosstabs(ct.generate_crosstabs(data, CATEGORICAL_VARS))


if __name__ == "__main__":
    main(sys.argv)