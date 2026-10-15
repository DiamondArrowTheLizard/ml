class Describe:
    def __init__(self, name: str, data):
        self.name = name
        self.data = data

    def __str__(self):
        return f"Describe: {self.name}\n{self.data}"


def generate_describes(dataframe) -> list:
    return [Describe(col, dataframe[col].describe()) for col in dataframe.columns]