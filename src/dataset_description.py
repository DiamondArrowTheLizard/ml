import pandas as pd


class DatasetDescription:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.shape = dataframe.shape
        self.columns = list(dataframe.columns)
        self.dtypes = dataframe.dtypes
        self.missing_values = dataframe.isnull().sum()

        self.numeric_columns = list(dataframe.select_dtypes(include="number").columns)
        self.categorical_columns = list(
            dataframe.select_dtypes(include=["object", "category"]).columns
        )

        self.numeric_summary = dataframe[self.numeric_columns].describe().T
        self.categorical_summary = {
            col: dataframe[col].value_counts() for col in self.categorical_columns
        }
        self.correlation_matrix = dataframe[self.numeric_columns].corr()

    def __str__(self):
        lines = []
        lines.append("=" * 60)
        lines.append("DATASET DESCRIPTION")
        lines.append("=" * 60)
        lines.append(f"Shape: {self.shape[0]} rows x {self.shape[1]} columns")
        lines.append("")
        lines.append("Columns and dtypes:")
        for col, dtype in self.dtypes.items():
            lines.append(f"  - {col}: {dtype}")

        lines.append("")
        lines.append("Missing values per column:")
        for col, count in self.missing_values.items():
            lines.append(f"  - {col}: {count}")

        lines.append("")
        lines.append("Numeric features summary:")
        lines.append(str(self.numeric_summary))

        lines.append("")
        lines.append("Categorical features value counts:")
        for col, counts in self.categorical_summary.items():
            lines.append(f"  {col}:")
            lines.append(str(counts.to_string()))
            lines.append("")

        lines.append("Correlation matrix (numeric features):")
        lines.append(str(self.correlation_matrix))
        return "\n".join(lines)
