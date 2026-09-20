# Titanic Survival Prediction

A complete machine learning classification project for predicting whether a passenger survived the Titanic disaster.

The project covers the full machine learning workflow: exploratory data analysis, feature engineering, preprocessing, model comparison, cross-validation, feature ablation, final model training, prediction, and Kaggle submission generation.

---

## Table of Contents

- [Project Overview](#project-overview)

- [Objectives](#objectives)

- [Dataset](#dataset)

- [Project Architecture](#project-architecture)

- [Project Structure](#project-structure)

- [Machine Learning Pipeline](#machine-learning-pipeline)

- [Exploratory Data Analysis](#exploratory-data-analysis)

- [Feature Engineering](#feature-engineering)

- [Feature Selection](#feature-selection)

- [Data Preprocessing](#data-preprocessing)

- [Models Evaluated](#models-evaluated)

- [Model Evaluation](#model-evaluation)

- [Final Model](#final-model)

- [Training Pipeline](#training-pipeline)

- [Prediction Pipeline](#prediction-pipeline)

- [Kaggle Submission](#kaggle-submission)

- [Installation](#installation)

- [Usage](#usage)

- [Results](#results)

- [Key Learning Outcomes](#key-learning-outcomes)

- [Limitations](#limitations)

- [Future Improvements](#future-improvements)

---

# Project Overview

The Titanic Survival Prediction project uses supervised machine learning to predict whether a passenger survived the Titanic disaster.

The project uses passenger information such as:

- Passenger class

- Sex

- Age

- Number of siblings/spouses aboard

- Number of parents/children aboard

- Fare

- Port of embarkation

- Passenger title

- Ticket group size

- Cabin availability

The main objective was not only to build a classifier, but to implement a complete and reproducible machine learning workflow.

The project therefore includes:

1. Exploratory data analysis

2. Missing-value analysis

3. Feature engineering

4. Feature ablation

5. Data preprocessing

6. Multiple classification algorithms

7. Hyperparameter experiments

8. Leakage-safe cross-validation

9. Held-out validation

10. Final model training

11. Model serialization

12. Prediction on unseen data

13. Kaggle submission generation

---

# Objectives

The main objectives of the project are:

- Understand the Titanic dataset.

- Identify important patterns in passenger survival.

- Handle missing data correctly.

- Engineer useful features from raw passenger information.

- Compare different classification algorithms.

- Use cross-validation to estimate model performance.

- Avoid preprocessing data leakage.

- Select a practical final model.

- Build a reusable training and prediction pipeline.

- Generate predictions for unseen Kaggle test data.

---

# Dataset

The project uses the Titanic dataset from the Kaggle Titanic competition.

## Training Dataset

The training dataset contains:

- 891 passengers

- 12 original columns

The target variable is:


Survived


where:


0 = Did not survive

1 = Survived


### Original Features

| Feature     | Description                       |

| ----------- | --------------------------------- |

| PassengerId | Unique passenger identifier       |

| Survived    | Target variable                   |

| Pclass      | Passenger class                   |

| Name        | Passenger name                    |

| Sex         | Passenger sex                     |

| Age         | Passenger age                     |

| SibSp       | Number of siblings/spouses aboard |

| Parch       | Number of parents/children aboard |

| Ticket      | Ticket number                     |

| Fare        | Passenger fare                    |

| Cabin       | Cabin information                 |

| Embarked    | Port of embarkation               |

## Test Dataset

The Kaggle test dataset contains:


418 passengers


and does not contain the Survived column.

The test dataset is used only after the model and feature pipeline have been finalized.

---

# Project Architecture

The project follows this overall architecture:


                    Titanic Dataset

                          │

              ┌───────────┴───────────┐

              │                       │

         train.csv                test.csv

              │                       │

              ▼                       │

       Exploratory Analysis           │

              │                       │

              ▼                       │

       Feature Engineering             │

              │                       │

              ▼                       │

      Train / Validation Split          │

              │                       │

              ▼                       │

       Preprocessing Pipeline           │

              │                       │

              ▼                       │

       Model Experimentation            │

              │                       │

              ├── Logistic Regression   │

              ├── Decision Tree         │

              ├── Random Forest         │

              └── KNN                   │

              │                         │

              ▼                         │

       Cross-Validation                 │

              │                         │

              ▼                         │

       Feature Ablation                 │

              │                         │

              ▼                         │

        Final Model                     │

              │                         │

              ▼                         │

         train.py                       │

              │                         │

              ▼                         │

      Saved ML Pipeline                 │

              │                         │

              ├───────────────┐         │

              │               │         │

              ▼               ▼         ▼

        predict.py     create_submission.py

                              │

                              ▼

                       submission.csv


---

# Project Structure


titanic-survival-prediction/

│

├── data/

│   ├── train.csv

│   └── test.csv

│

├── models/

│   └── titanic_logistic_regression.joblib

│

├── notebooks/

│   └── 01_titanic_exploration.ipynb

│

├── src/

│   ├── train.py

│   ├── predict.py

│   └── create_submission.py

│

├── .gitignore

├── requirements.txt

├── README.md

└── submission.csv


### data/

Contains the Kaggle Titanic datasets.

The datasets are excluded from version control.

### models/

Contains generated trained model artifacts.

The .joblib model is excluded from version control.

### notebooks/

Contains the exploratory and experimental machine learning notebook.

01_titanic_exploration.ipynb contains:

* Dataset inspection

* Missing-value analysis

* Exploratory analysis

* Feature engineering experiments

* Model experiments

* Cross-validation

* Feature ablation

* Validation evaluation

### src/train.py

Responsible for:

* Loading the training dataset

* Creating engineered features

* Building the preprocessing pipeline

* Building the Logistic Regression model

* Training the complete pipeline

* Saving the trained pipeline

### src/predict.py

Responsible for:

* Loading the saved model

* Preparing passenger data

* Applying feature engineering

* Generating a prediction

* Returning survival probabilities

### src/create_submission.py

Responsible for:

* Loading Kaggle's test dataset

* Loading the trained model

* Generating predictions

* Creating submission.csv

---

# Machine Learning Pipeline

The final machine learning pipeline is:


Raw Passenger Data

        │

        ▼

Feature Engineering

        │

        ├── Title

        ├── TicketGroupSize

        └── CabinKnown

        │

        ▼

Column Selection

        │

        ├── Numerical Features

        │

        └── Categorical Features

        │

        ▼

Preprocessing

        │

        ├── Numerical

        │     └── Median Imputation

        │

        └── Categorical

              ├── Most-Frequent Imputation

              └── One-Hot Encoding

        │

        ▼

Logistic Regression

        │

        ▼

Prediction


The preprocessing and classifier are stored together inside a scikit-learn Pipeline.

This allows the exact same transformations to be applied during training and prediction.

---

# Exploratory Data Analysis

The exploratory analysis investigated:

* Dataset dimensions

* Data types

* Missing values

* Target distribution

* Categorical feature distributions

* Numerical feature distributions

* Survival rates across categorical variables

* Relationships between passenger characteristics and survival

## Missing Values

Important missing-value counts in the training dataset were:

| Feature  | Missing Values | Missing Percentage |

| -------- | -------------: | -----------------: |

| Age      |            177 |             19.87% |

| Cabin    |            687 |             77.10% |

| Embarked |              2 |              0.22% |

The large number of missing cabin values motivated the creation of a simpler CabinKnown feature rather than using the raw cabin identifier directly.

## Target Distribution

The training dataset contains approximately:


61.6% did not survive

38.4% survived


A majority-class classifier therefore provides a baseline accuracy of approximately:


61.45%


---

# Feature Engineering

Several features were engineered from the original data.

## Title

Passenger titles were extracted from the Name column.

For example:


Smith, Mrs. John


produces:


Mrs


The title captures information that is not directly represented by the raw name.

The original title representation was retained after experimentation.

A normalized-title experiment was also performed, but it produced a lower cross-validation score and was therefore not used in the final pipeline.

---

## TicketGroupSize

The number of passengers sharing the same ticket was calculated using:


df["TicketGroupSize"] = (

    df.groupby("Ticket")["Ticket"].transform("count")

)


This converts the raw ticket identifier into a numerical feature describing the size of the associated passenger group.

---

## CabinKnown

A binary feature was created indicating whether cabin information was available:


df["CabinKnown"] = df["Cabin"].notna().astype(int)


The feature therefore takes values:


0 = Cabin information unavailable

1 = Cabin information available


---

## FamilySize

Family size was initially investigated using:


FamilySize = SibSp + Parch + 1


However, feature ablation experiments did not show a meaningful improvement from this feature in the final model configuration, so it was excluded from the final model.

---

# Feature Selection

Feature ablation was used to determine whether engineered features provided measurable improvements.

The main results were:

| Feature Set                                  | Mean CV Accuracy |

| -------------------------------------------- | ---------------: |

| Basic features                               |           79.63% |

| Basic + Title                                |           81.74% |

| Basic + Title + CabinKnown                   |           82.73% |

| Basic + Title + TicketGroupSize + CabinKnown |           82.87% |

| Basic + Title + FamilySize + CabinKnown      |           82.59% |

| Full feature set                             |           82.87% |

The final feature set was therefore:

### Numerical Features


Pclass

Age

SibSp

Parch

Fare

CabinKnown

TicketGroupSize


### Categorical Features


Sex

Embarked

Title


FamilySize was excluded from the final model.

---

# Data Preprocessing

Preprocessing was implemented using scikit-learn's ColumnTransformer and Pipeline.

## Numerical Features

Numerical features use median imputation:


SimpleImputer(strategy="median")


This handles missing numerical values such as Age.

## Categorical Features

Categorical features use:


SimpleImputer(strategy="most_frequent")


followed by:


OneHotEncoder(handle_unknown="ignore")


The handle_unknown="ignore" setting ensures that an unseen category during prediction does not cause the pipeline to fail.

---

# Preventing Data Leakage

A major part of the project was ensuring that preprocessing did not leak information from validation data into training.

An incorrect approach would be:


Preprocess entire dataset

        ↓

Create CV folds

        ↓

Train and validate


In this approach, preprocessing has already seen the entire dataset before cross-validation.

Instead, the project uses:


Raw Training Data

       │

       ▼

Cross-Validation

       │

       ├── Fold Training Data

       │       │

       │       ▼

       │   Fit Preprocessor

       │       │

       │       ▼

       │   Train Model

       │

       └── Fold Validation Data

               │

               ▼

        Transform using the

        fitted preprocessor

               │

               ▼

            Predict


The preprocessing steps are therefore fitted independently inside each cross-validation fold.

This was implemented using:


Pipeline([

    ("preprocessor", preprocessor),

    ("model", classifier)

])


This makes the cross-validation estimates more methodologically reliable.

---

# Models Evaluated

Four classification algorithms were evaluated.

## 1. Logistic Regression

Configuration:


LogisticRegression(

    max_iter=1000,

    random_state=42

)


## 2. Decision Tree

Configuration:


DecisionTreeClassifier(

    max_depth=4,

    random_state=42

)


The maximum depth was selected after experimenting with tree depths from 1 to 10.

## 3. Random Forest

Configuration:


RandomForestClassifier(

    n_estimators=100,

    random_state=42

)


Different numbers of trees were tested:


50

100

200

300


Increasing the number of trees beyond 100 did not improve validation performance in the experiment.

## 4. K-Nearest Neighbors

Configuration:


KNeighborsClassifier(

    n_neighbors=5

)


Different values of k were investigated.

Scaling was included for KNN because it is a distance-based algorithm.

---

# Model Evaluation

The models were evaluated using:

* Accuracy

* Precision

* Recall

* F1-score

* Confusion matrix

* 5-fold stratified cross-validation

The training data was divided into:


80% training

20% held-out validation


The split used stratification to preserve the target class distribution.

---

# Cross-Validation Results

The leakage-safe 5-fold cross-validation results for the four model configurations were:

| Model               | Mean CV Accuracy | Standard Deviation |

| ------------------- | ---------------: | -----------------: |

| Logistic Regression |           82.87% |              2.72% |

| Random Forest       |           81.47% |              1.58% |

| KNN                 |           80.76% |              1.56% |

| Decision Tree       |           80.34% |              1.36% |

The standard deviation represents variation in accuracy across the five folds.

---

# Validation Results

The models were also evaluated on the held-out validation set.

| Model               | Validation Accuracy |

| ------------------- | ------------------: |

| Logistic Regression |              83.80% |

| Decision Tree       |              82.12% |

| KNN                 |              81.56% |

| Random Forest       |              79.33% |

### Logistic Regression


Validation Accuracy: 83.80%

              precision    recall  f1-score   support

           0       0.86      0.88      0.87       110

           1       0.80      0.77      0.79        69

    accuracy                           0.84       179


Confusion matrix:


[[97, 13],

 [16, 53]]


---

# Final Model

The final model is a Logistic Regression classifier combined with a preprocessing pipeline.

The final feature set is:


Numerical:

\- Pclass

\- Age

\- SibSp

\- Parch

\- Fare

\- CabinKnown

\- TicketGroupSize

Categorical:

\- Sex

\- Embarked

\- Title


The final candidate pipeline achieved:


Cross-validation accuracy: 82.87%

Cross-validation standard deviation: 2.72%

Validation accuracy: 83.80%


Validation classification performance:


Survivor precision: 79%

Survivor recall:    78%

Survivor F1-score:  79%


Validation confusion matrix:


[[96, 14],

 [15, 54]]


The final pipeline was then trained on all 891 labeled training examples before generating predictions for the Kaggle test set.

---

# Training Pipeline

src/train.py implements the final training workflow.


train.csv

   │

   ▼

Load Data

   │

   ▼

Feature Engineering

   │

   ├── Title

   ├── TicketGroupSize

   └── CabinKnown

   │

   ▼

Separate X and y

   │

   ▼

ColumnTransformer

   │

   ├── Numerical Pipeline

   │     └── Median Imputation

   │

   └── Categorical Pipeline

         ├── Most-Frequent Imputation

         └── One-Hot Encoding

   │

   ▼

Logistic Regression

   │

   ▼

Fit on 891 training samples

   │

   ▼

Save complete pipeline

   │

   ▼

titanic_logistic_regression.joblib


The saved .joblib file contains the fitted preprocessing and model components together.

---

# Prediction Pipeline

src/predict.py loads the saved pipeline and applies it to new passenger data.


New Passenger

      │

      ▼

Create DataFrame

      │

      ▼

Feature Engineering

      │

      ▼

Saved Preprocessor

      │

      ▼

Logistic Regression

      │

      ▼

Prediction

      │

      ├── Survived

      └── Did not survive


The pipeline also provides class probabilities using:


model.predict_proba()


---

# Kaggle Submission

src/create_submission.py generates predictions for the 418 passengers in test.csv.

The resulting file contains exactly two columns:


PassengerId

Survived


The generated submission was structurally validated:


Rows: 418

Columns: 2

Missing values: 0

Unique PassengerIds: 418

Prediction classes: 0, 1


The prediction distribution was:


Did not survive: 253

Survived:        165


The generated file is:


submission.csv


The submission file is ignored by Git because it is a generated artifact.

---

# Installation

Clone the repository and navigate to the project directory.

Create a virtual environment:


python -m venv .venv


Activate it on Windows:


.venv\Scripts\activate


Install the dependencies:


pip install -r requirements.txt


---

# Usage

## 1. Train the Model


python src/train.py


This:

* Loads data/train.csv

* Creates engineered features

* Builds the preprocessing pipeline

* Trains Logistic Regression

* Saves the fitted pipeline to:


models/titanic_logistic_regression.joblib


---

## 2. Make an Individual Prediction


python src/predict.py


This loads the saved model and demonstrates prediction on a passenger.

The output includes:


Prediction

Probability of not surviving

Probability of surviving


---

## 3. Generate Kaggle Submission


python src/create_submission.py


This loads:


data/test.csv


and generates:


submission.csv


---

# Git Workflow

The project was developed incrementally using Git and GitHub.

Major checkpoints included:


Initial project setup

        ↓

EDA and preprocessing

        ↓

Model experiments

        ↓

Model comparison

        ↓

Feature engineering and ablation

        ↓

Final model

        ↓

train.py

        ↓

predict.py

        ↓

create_submission.py


Generated files such as datasets, trained model artifacts, and the Kaggle submission are excluded from version control.

---

# Key Learning Outcomes

This project provided practical experience with:

### Data Analysis

* Pandas

* Dataset inspection

* Missing-value analysis

* Categorical analysis

* Numerical feature analysis

* Group-based analysis

### Feature Engineering

* Extracting information from strings

* Group-based features

* Binary indicator features

* Feature ablation

* Comparing engineered features quantitatively

### Machine Learning

* Logistic Regression

* Decision Trees

* Random Forest

* K-Nearest Neighbors

* Hyperparameter experiments

* Cross-validation

* Train/validation splits

### Model Evaluation

* Accuracy

* Precision

* Recall

* F1-score

* Confusion matrices

* Cross-validation mean

* Cross-validation standard deviation

### ML Engineering

* Pipeline

* ColumnTransformer

* SimpleImputer

* OneHotEncoder

* Model serialization with Joblib

* Reusable training scripts

* Reusable prediction scripts

* Automated submission generation

### Reproducibility

The final project separates:


Experimentation

      ↓

Final feature configuration

      ↓

Reusable training pipeline

      ↓

Saved model

      ↓

Prediction pipeline


This makes the project reproducible rather than dependent on manually executed notebook cells.

---

# Limitations

This project has several limitations.

* The dataset is relatively small, containing only 891 labeled training examples.

* Results depend on the particular train/validation split and dataset.

* The selected model was not extensively optimized with large-scale hyperparameter search.

* The Titanic dataset contains historical passenger information with substantial missing data.

* Accuracy alone does not fully describe classification performance.

* The Kaggle test set does not provide labels locally, so its true performance cannot be evaluated before submission.

The reported validation and cross-validation results should therefore be interpreted as estimates for this dataset rather than general performance guarantees.

---

# Future Improvements

Potential future extensions include:

* More systematic hyperparameter optimization

* Additional feature engineering

* Title normalization experiments using alternative representations

* More sophisticated handling of cabin information

* Family-level feature engineering

* Ensemble models

* Gradient boosting methods

* Calibration analysis

* Threshold analysis

* Deployment as a web API

* Interactive frontend for passenger predictions

---

# Conclusion

This project implements an end-to-end Titanic survival classification workflow.

Starting from raw passenger data, the project performs exploratory analysis, feature engineering, leakage-safe preprocessing, model comparison, feature ablation, final model training, and prediction on unseen Kaggle data.

The final implementation packages preprocessing and Logistic Regression into a reusable scikit-learn pipeline, allowing the same transformations to be consistently applied during training and prediction.

The project therefore demonstrates both the machine learning concepts and the engineering practices required to turn an experimental notebook into a reproducible ML project.