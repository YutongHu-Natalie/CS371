import pandas as pd
import re
import numpy as np
from openai import OpenAI

# Initialize OpenAI client with your API key
client = OpenAI(api_key="key")


# List of valid topics
valid_topics = [
    "Economy", "Healthcare", "Education", "Immigration", "Climate",
    "Foreign Policy", "Social Justice", "Infrastructure", "Gun Control",
    "Government Policy", "Veterans Affairs", "Technology"
]

def extract_topics(text):
    """Extract topics using the OpenAI Chat API."""
    try:
        # Increased chunk size to reduce API calls
        chunks = list(chunk_text_by_period(text, max_sentences=30))

        for chunk in chunks:
            prompt = (
                f"The following are sentences said by Donald Trump. Extract key topics that align with "
                f"predefined categories relevant to public policy and political discourse:\n\n"
                f"{chunk}\n\n"
                "Select only from the following predefined categories: Economy, Healthcare, Education, "
                "Immigration, Climate, Foreign Policy, Social Justice, Infrastructure, Gun Control, "
                "Government Policy, Veterans Affairs, Technology.\n\n"
                "If no topics are detected, respond with 'NaN'. Your response should contain only the relevant topics, "
                "separated by commas. Do not add extra explanations."
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
                top_p=0.9
            )

            result = response.choices[0].message.content.strip()
            print(f"API Response: {result}")  # Debugging

            # Extract topics directly by splitting
            topics = [t.strip() for t in result.split(',') if t.strip() in valid_topics]

            if topics:
                return ', '.join(topics)  # Return first valid set of topics

        return np.nan  # If no valid topics found

    except Exception as e:
        print(f"Error during topic extraction: {e}")
        return np.nan

def chunk_text_by_period(text, max_sentences=10):
    """Split text into chunks with max number of sentences."""
    sentences = re.split(r'\.\s+', text)
    chunks = [' '.join(sentences[i:i + max_sentences]) for i in range(0, len(sentences), max_sentences)]
    return chunks

# Load the cleaned transcripts dataset
df = pd.read_csv('trump_transcripts_cleaned.csv')
# df = df.iloc[:2]  # Limit the number of rows for testing
df = df[df['cleaned'].notna()]  # Ensure non-empty transcripts

# Apply the topic extraction function
df['topic'] = df['cleaned'].apply(extract_topics)


# Drop rows where no topics were detected
df = df.dropna(subset=['topic'])

# Display the DataFrame
print(df[['cleaned', 'Date', 'Title', 'URL', 'topic']])

# Save the results to a CSV file
df.to_csv('trump_transcripts_with_topics (v2).csv', index=False)