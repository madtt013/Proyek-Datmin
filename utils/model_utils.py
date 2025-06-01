import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH = "data/healthcare-dataset-stroke-data.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

def train_model():
    df = load_data()
    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    joblib.dump(model, "model.pkl")

    return accuracy, report

def load_model():
    return joblib.load("model.pkl")

