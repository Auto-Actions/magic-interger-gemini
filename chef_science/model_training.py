import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class ModelTraining:
    def __init__(self):
        self.model = None

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains a logistic regression model.

        Args:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels.
        """
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
