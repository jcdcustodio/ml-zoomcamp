import re
import pickle

import numpy as np
import pandas as pd
import sklearn.pipeline

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

SEED = 1

# ############################
# 01 - LOAD AND TRANSFORM
# ############################

DATA_PATH = Path("../datasets/telco_customer_churn.csv").resolve()

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Normalize column names
    df.columns = df.rename(columns=normalize_name).columns

    # Fix format in each categorical variables
    categorical_columns = list(df.dtypes[df.dtypes == "str"].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    # Fill NULLs with zero in `total_charges`
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)

    # Convert `churn` into binary values
    df["churn"] = (df["churn"] == "yes").astype(int)

    return df


def normalize_name(s: str) -> str:
    # Insert underscore between consecutive capitals followed by lowercase 
    # (e.g., 'XMLHttp' -> 'XML_Http')
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    # Insert underscore between lowercase/numbers and following uppercase 
    # (e.g., 'HttpReq' -> 'Http_Req')
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)

    return s.lower()


# #####################
# 02 - TRAIN MODEL
# #####################

def train_model(df: pd.DataFrame) -> sklearn.pipeline.Pipeline:

    # Sort features
    numerical_features = ["tenure", "monthly_charges", "total_charges"]

    exclude_cols = numerical_features[:]
    exclude_cols.extend(["churn", "customer_id"])
    categorical_features = df.columns.difference(exclude_cols, sort=False).to_list()

    all_features = categorical_features + numerical_features

    churn_model = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(
            solver="liblinear", 
            C=1.0, 
            max_iter=1000, 
            random_state=SEED
        )
    )

    train_dict = df[all_features].to_dict(orient="records")
    y_full_train = df["churn"].values

    churn_model.fit(train_dict, y_full_train)

    return churn_model


# ####################
# 03 - SAVE MODEL
# ####################

def save_model(model, filename="model.bin"):
    with open(filename, "wb") as f_out:
        pickle.dump(model, f_out)

    print(f"Model file successfully saved to {Path(__file__).parent.resolve()}/{filename}")


df = load_data()
model_pipeline = train_model(df)
save_model(model_pipeline, filename="model_final.bin")
