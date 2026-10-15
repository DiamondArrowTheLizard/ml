from itertools import combinations

import pandas as pd


class Crosstab:
    def __init__(self, var1: str, var2: str, data):
        self.var1 = var1
        self.var2 = var2
        self.data = data

    def __str__(self):
        return f"Crosstab: {self.var1} vs {self.var2}\n{self.data}"


def generate_crosstabs(dataframe, categorical_vars: list) -> list:
    crosstab_list = []
    for var1, var2 in combinations(categorical_vars, 2):
        crosstab = Crosstab(var1, var2, pd.crosstab(dataframe[var1], dataframe[var2]))
        crosstab_list.append(crosstab)
    return crosstab_list


def print_crosstab(crosstab_list: list, idx: int):
    crosstab = crosstab_list[idx]
    print(crosstab)


def print_crosstabs(crosstab_list: list):
    for i in range(len(crosstab_list)):
        print_crosstab(crosstab_list, i)