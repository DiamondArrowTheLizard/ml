import sys

import csv_to_dataframe as ctd


def main(argv):
    if len(argv) < 2:
        print("No .csv specified.")
        return

    data = ctd.csv_to_dataframe(argv[1])
    print(data)


if __name__ == "__main__":
    main(sys.argv)
