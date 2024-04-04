import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

class ModelEvaluation:
    def __init__(self):
        pass

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate the performance of a trained model on a test set.

        Args:
            model: The trained model.
            X_test: The test data features.
            y_test: The test data labels.

        Returns:
            A dictionary containing the evaluation metrics.
        """

        # Make predictions on the test set.
        y_pred = model.predict(X_test)

        # Calculate the accuracy.
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate the classification report.
        classification_report = classification_report(y_test, y_pred)

        # Return the evaluation metrics.
        return {
            "accuracy": accuracy,
            "classification_report": classification_report,
        }
