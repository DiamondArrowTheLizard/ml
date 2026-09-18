import pandas as pd


def csv_to_dataframe(csv_filename):
    return pd.read_csv(csv_filename)
