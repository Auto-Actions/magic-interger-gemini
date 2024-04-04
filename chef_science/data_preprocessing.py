import pandas as pd

class DataPreprocessing:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    def load_data(self, path: str):
        self.data = pd.read_csv(path)

    def clean_data(self):
        # Implement data cleaning logic
        self.data.dropna(inplace=True)
        self.data.fillna(self.data.mean(), inplace=True)

    def split_data(self, test_size: float = 0.2):
        # Implement data splitting logic
        from sklearn.model_selection import train_test_split
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, y_train, X_test, y_test
