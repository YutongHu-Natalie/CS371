import pandas as pd
import numpy as np
import re

data = pd.read_csv('trump_transcripts_cleaned.csv')
print(data.info())

df_filtered = data[~data['Title'].str.contains("debate", case=False, na=False)]
print(df_filtered.info())

df_filtered1 = df_filtered[["Title", "Date", "Transcript", "URL"]]
print(df_filtered1.info())