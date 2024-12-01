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

    precision_scores = []

    for column in features.columns:
        weighted = precision_score(target, features[column], average="weighted", zero_division=0)
        macro = precision_score(target, features[column], average="macro", zero_division=0)
        precision_scores.append((column, round(weighted, 3), round(macro, 3)))

    precision_df = pd.DataFrame(precision_scores, columns=["Column", "Weighted Precision", "Macro Precision"])
    print(precision_df)


print("Harris")
calculate_precision(harris)
print("\n")
print("----------------")
print("Trump")
calculate_precision(trump)