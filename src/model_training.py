from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib

def load_data(train_path):
    data = pd.read_csv(train_path)
    return data

def preprocess_data(data):
    # Assuming 'Consumer disputed' is the target variable
    X = data.drop('Consumer disputed', axis=1)
    y = data['Consumer disputed']
    
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    return model

def save_model(model, model_path):
    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_path = "C:\\Users\\KannanRamaswamy\\Downloads\\Edv-Proj1\\Data\\Consumer_Complaints_train.csv"
    model_path = "models/trained_model.pkl"
    
    data = load_data(train_path)
    X, y = preprocess_data(data)
    model = train_model(X, y)
    save_model(model, model_path)