import pandas as pd
import numpy as np
from datashader import where

df= pd.read_csv("harris_claude_output.csv")
subset_col= ["Statement Classification", "Statement Ideology Score", "Processed"]
df.dropna(subset= subset_col, inplace=True)
df.drop()







