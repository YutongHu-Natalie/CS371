import pandas as pd
import numpy as np
import os

harris_haiku = pd.array([2,0,0,0,2,0,4,2,0,0,0,0,0,4,0],dtype = "Int64")
harris_sonnet = pd.array([0,0,0,0,2,0,4,3,0,0,0,0,0,4,0],dtype = "Int64")
harris_4o = pd.array([0,0,0,0,3,0,4,3,0,0,0,0,0,4,0],dtype = "Int64")
harris_mini = pd.array([0,0,0,0,2,2,3,2,0,0,0,0,0,4,2],dtype = "Int64")
harris_1 = pd.array([0,0,0,0,2,0,0,3,0,0,0,0,0,4,0],dtype = "Int64")

harris = pd.DataFrame({
    "Haiku": harris_haiku,
    "Sonnet":harris_sonnet,
    "4o": harris_4o,
    "4o mini": harris_mini,
    "human 1": harris_1
})

print(harris)

trump_haiku = pd.array([2,0,2,0,0,2,4,2,2,2,2,2,4,4,2],dtype = "Int64")
trump_sonnet = pd.array([2,0,2,0,0,2,4,4,0,4,0,0,4,4,2],dtype = "Int64")
trump_4o = pd.array([0,0,0,0,0,2,4,0,0,1,0,0,4,4,2],dtype = "Int64")
trump_mini = pd.array([1,0,1,0,0,1,4,3,3,1,1,0,4,4,2],dtype = "Int64")
trump_1 = pd.array([0,0,0,0,0,2,4,0,0,4,0,0,4,4,2],dtype = "Int64")

trump = pd.DataFrame({
    "Haiku": trump_haiku,
    "Sonnet":trump_sonnet,
    "4o": trump_4o,
    "4o mini": trump_mini,
    "human 1": trump_1    
})

print(trump)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
path = os.path.join(root_dir, "CS371", "Results")

harris.to_csv(os.path.join(path, "harris_results.csv"), index=True)
trump.to_csv(os.path.join(path, "trump_results.csv"), index=True)