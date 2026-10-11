import os
import json
import joblib
import pandas as pd 
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
    )
from xgboost import XGBClassifier

# path 

INPUT_FILE = "data/reference.csv"
MODEL_DIR="models"

MODEL_FILE = os.path.join(MODEL_DIR,"credit_risk_model.pkl")

METADATA_FILE = os.path.join(MODEL_DIR,"model_metadata.json")


TARGET_COLUMN = "default_payment_next_month"

RANDOM_STATE =42

# Reference data loading 

print("loading reference data...")

df= pd.read_csv(INPUT_FILE)
print(f"Datasets shape: {df.shape}")

# separate features and target 

print("\n separating features and target...")

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )
X = df.drop(columns= [TARGET_COLUMN])

y = df[TARGET_COLUMN]

print (f"Number of features:{X.shape[1]}")
print (f"Number of sample :{X.shape[0]}")

print("\nFeatures")
print(X.columns.to_list())

print("\nTarget distribution")
print(y.value_counts())

