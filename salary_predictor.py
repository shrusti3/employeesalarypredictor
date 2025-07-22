# salary_predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Keep only the form-used features + target
    used_features = ['age', 'workclass', 'education', 'educational-num', 'marital-status', 
                     'occupation', 'relationship', 'race', 'gender', 
                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df = df[used_features]

    # Encode categorical columns
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                'relationship', 'race', 'gender', 'native-country']
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Encode target
    le_income = LabelEncoder()
    df['income'] = le_income.fit_transform(df['income'])
    label_encoders['income'] = le_income

    return df, label_encoders

def train_model(df):
    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return model

def save_artifacts(model, label_encoders):
    joblib.dump(model, 'salary_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

if __name__ == "__main__":
    df = load_data('adult_3.csv')
    df_processed, label_encoders = preprocess_data(df)
    model = train_model(df_processed)
    save_artifacts(model, label_encoders)
    print("🔥 Model & encoders saved — synced with form!")
