import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(df):
    # Pisahkan fitur dan target
    X = df.drop("stroke ", axis=1)  # Ganti "stroke" dengan nama kolom target
    y = df["stroke "]  # Ganti "stroke" dengan nama kolom target
    
    # Konversi variabel kategorikal ke numerik
    X = pd.get_dummies(X, drop_first=True)
    
    # Bagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Latih model RandomForest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return report
