import data_preprocessing as dp
import model_training as mt
import model_evaluation as me

def main():
    # Load and preprocess data
    data_preprocessor = dp.DataPreprocessing()
    data_preprocessor.load_data("data.csv")
    data_preprocessor.clean_data()
    X_train, y_train, X_test, y_test = data_preprocessor.split_data(test_size=0.2)

    # Train model
    model_trainer = mt.ModelTraining()
    model = model_trainer.train_model(X_train, y_train)

    # Evaluate model
    model_evaluator = me.ModelEvaluation()
    model_evaluator.evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
