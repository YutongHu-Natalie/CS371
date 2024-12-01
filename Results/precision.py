import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
path = os.path.join(root_dir, "CS371", "Results")

harris = pd.read_csv(os.path.join(path, "harris_results.csv"))
trump = pd.read_csv(os.path.join(path, "trump_results.csv"))



def calculate_precision(df):
    df = df.iloc[:, 1:]
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]

    precision_scores = {}

    for column in features.columns:
        precision = precision_score(
        target, features[column], average="weighted", zero_division=0
    )
        precision = round(precision, 3)
        precision_scores[column] = precision

    precision_df = pd.DataFrame(
    list(precision_scores.items()), columns=["Column", "Precision"]
)
    print(precision_df)

calculate_precision(harris)
print("\n")
print("----------------")
calculate_precision(trump)