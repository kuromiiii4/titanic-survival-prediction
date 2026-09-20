from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "titanic_logistic_regression.joblib"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

def create_features(df):
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r", ([^.]*)\.")
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    df["CabinKnown"] = df["Cabin"].notna().astype(int)

    return df

def main():

    #Load the saved model
    model = joblib.load(MODEL_PATH)

    #Load Kaggle test data
    test_df = pd.read_csv(TEST_PATH)

    #Create the same features used during training
    test_features = create_features(test_df)

    #Generate predictions
    predictions = model.predict(test_features)

    #create Kaggle submission
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictions
    })

    #Save submission
    submission.to_csv(SUBMISSION_PATH, index = False)

    print("Submission created.")
    print(f"Rows: {len(submission)}")
    print(f"Saved to: {SUBMISSION_PATH}")
    print()
    print("Prediction distribution:")
    print(submission["Survived"].value_counts())
    print()
    print("First 10 predictions:")
    print(submission.head(10))

if __name__ == "__main__":
    main()