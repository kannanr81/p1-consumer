import pandas as pd
from data_preprocessing import load_data, preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

def main():
    # Load and preprocess the training data
    train_data = load_data("data/Consumer_Complaints_train.csv")
    X_train, y_train = preprocess_data(train_data, target_column="Consumer disputed")

    # Train the prediction model
    model = train_model(X_train, y_train)

    # Load and preprocess the test data
    test_data = load_data("data/Consumer_Complaints_test_share.csv")
    X_test, y_test = preprocess_data(test_data, target_column="Consumer disputed")

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()