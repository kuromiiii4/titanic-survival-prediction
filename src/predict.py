from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "titanic_logistic_regression.joblib"

def create_features(df):
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r", ([^.]*)\.")
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    df["CabinKnown"] = df["Cabin"].notna().astype(int)

    return df

def predict_survival(passenger_data):
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([passenger_data])
    df = create_features(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    return prediction, probability

if __name__ == "__main__":

    passenger = {
        "Pclass": 1,
        "Name": "Smith, Mrs. John",
        "Sex": "female",
        "Age": 30,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "TEST123",
        "Fare": 100,
        "Cabin": "C85",
        "Embarked": "S"
    }

    prediction, probability = predict_survival(passenger)

    print(
        "Prediction:",
        "Survived" if prediction == 1 else "Did not survive"
    )

    print(f"Probability of not surviving: {probability[0]:.2%}")
    print(f"Probability of surviving: {probability[1]:.2%}")