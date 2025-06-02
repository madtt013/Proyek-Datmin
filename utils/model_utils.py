import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(df):
    df = df.dropna()
    X = df.drop(['id', 'stroke'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = df['stroke']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    joblib.dump(model, 'model.pkl')
    return report
