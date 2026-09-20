from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_DIR = PROJECT_ROOT /"models"
MODEL_PATH = MODEL_DIR / "titanic_logistic_regression.joblib"

# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def create_features(df):
    df = df.copy()

    #Extract passenger title
    df["Title"] = df["Name"].str.extract(r", ([^.]*)\.")

    #Number of passengers sharing the same ticket
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")

    #Whether cabin information is available
    df["CabinKnown"] = df["Cabin"].notna().astype(int)

    return df

# --------------------------------------------------
# Main training function
# --------------------------------------------------

def main():

    #Load dataset
    df = pd.read_csv(DATA_PATH)

    #Create engineered features
    df = create_features(df)

    #Separate featuers and target
    X = df.drop(columns = ["Survived"])
    y = df["Survived"]

    #Final feature configuration
    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "CabinKnown",
        "TicketGroupSize"
    ]

    categorical_features = [
        "Sex",
        "Embarked",
        "Title"
    ]

    #Numerical preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    #Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    #Combined preprocessing
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    #Complete ML Pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state = 42
        ))
    ])

    #Train on the complete labeled dataset
    model.fit(X, y)

    #Create model directory if necessary
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    #Save complete pipeline
    joblib.dump(model, MODEL_PATH)

    print("Training completed.")
    print(f"Training samples: {len(X)}")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ =="__main__":
    main()