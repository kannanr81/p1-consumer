import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Handle missing values
    data = data.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_data(train_file, target_column):
    data = load_data(train_file)
    cleaned_data, label_encoders = clean_data(data)
    X_train, X_test, y_train, y_test = split_data(cleaned_data, target_column)
    return X_train, X_test, y_train, y_test, label_encoders