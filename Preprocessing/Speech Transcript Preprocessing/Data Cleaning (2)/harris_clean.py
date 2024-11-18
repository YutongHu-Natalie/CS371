import pandas as pd
import numpy as np
import re

data = pd.read_csv('kamala_harris_transcripts.csv')
data = data.iloc[:142]
df_filtered = data[~data['Title'].str.contains("debate", case=False, na=False)]

df_filtered1 = df_filtered.copy()
df_filtered1['Transcript'] = df_filtered1['Transcript'].str[124:]
df_filtered1['Transcript'] = df_filtered1['Transcript'].str[:-555]

# Define relevant speakers
relevant_speakers = ["Kamala Harris", "Vice President Kamala Harris", "Kamala", "Harris",
                     "Senator Kamala Harris", "Vice President Harris", "VP Kamala Harris", "Madam VP Kamala Harris", "VP Harris"]

# Function to split, filter, and concatenate relevant speaker content with fallback to another pattern
def split_filter_concat_speaker(text, relevant_speakers):
    # Patterns to match both mm:ss and hh:mm:ss timestamps
    patterns = [
        r'([A-Za-z\s]+)\s*\(\d{1,2}:\d{2}(?::\d{2})?\):\s*(.*?)(?=[A-Za-z\s]+\s*\(\d{1,2}:\d{2}(?::\d{2})?\):|$)',
        r'([A-Za-z\s]+):\s*\(\d{1,2}:\d{2}(?::\d{2})?\)\s*(.*?)(?=(?:[A-Za-z\s]+:\s*\(\d{1,2}:\d{2}(?::\d{2})?\))|$)'
    ]
    concatenated_content = ""

    # Try each pattern until content is found
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Create DataFrame from matches
            speaker_df = pd.DataFrame(matches, columns=["Speaker", "Content"])
            speaker_df["Speaker"] = speaker_df["Speaker"].str.strip()
            speaker_df["Content"] = speaker_df["Content"].str.strip()

            # Concatenate consecutive entries for relevant speakers
            last_speaker = None
            temp_content = ""
            for _, row in speaker_df.iterrows():
                if row["Speaker"] in relevant_speakers:
                    if row["Speaker"] == last_speaker:
                        temp_content += " " + row["Content"]
                    else:
                        concatenated_content += " " + temp_content if temp_content else ""
                        temp_content = row["Content"]
                        last_speaker = row["Speaker"]
            concatenated_content += " " + temp_content if temp_content else ""
            break

    # Clean up whitespace
    return re.sub(r'\s+', ' ', concatenated_content).strip()

# Sample DataFrame
df_filtered2 = df_filtered1.copy()

# Replacements for generic speakers
replacements = {11: "Speaker 1", 52: "Speaker 3", 41: "Speaker 3", 62: "Speaker 2"}
for index, speaker in replacements.items():
    df_filtered2.at[index, "Transcript"] = df_filtered2.at[index, "Transcript"].replace(speaker, "Harris")

# Apply the function to each cell in 'Transcript' and store results in 'cleaned' column
df_filtered2['cleaned'] = df_filtered2['Transcript'].apply(lambda x: split_filter_concat_speaker(x, relevant_speakers))
df_filtered2 = df_filtered2[df_filtered2['cleaned'] != ""].reset_index(drop=True)
# Display sample to verify
df_filtered2[['Transcript', 'cleaned']].iloc[0:140]

# Assuming df_filtered2 is defined, display rows where the 'cleaned' column contains empty strings
empty_strings_df = df_filtered2[df_filtered2['cleaned'] == ""]
empty_strings_df[['Title', 'cleaned']]

# Drop rows where 'cleaned' column has empty strings
df_filtered2 = df_filtered2[df_filtered2['cleaned'] != ""]

print(df_filtered2.info())

df_filtered2.to_csv('harris_transcripts_cleaned.csv', index=False)