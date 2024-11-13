import anthropic
import pandas as pd
import re
import numpy as np
import os
from anthropic import Anthropic
from typing import List, Optional

client = anthropic.Anthropic(api_key="key")

VALID_TOPICS = {
    "Economy", "Healthcare", "Education", "Immigration", "Climate",
    "Foreign Policy", "Social Justice", "Infrastructure", "Gun Control",
    "Government Policy", "Veterans Affairs", "Technology"
}

def chunk_text_by_period(text: str, max_sentences: int = 40) -> List[str]:  # Increased from 10 to 30
    """Split text into chunks with a maximum number of sentences."""
    if not isinstance(text, str):
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [' '.join(sentences[i:i + max_sentences]) for i in range(0, len(sentences), max_sentences)]

def extract_topics(text: str) -> Optional[str]:
    """Extract topics from text using the Claude API."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    
    try:
        chunks = chunk_text_by_period(text)
        all_topics = set()
        
        for chunk in chunks:
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                temperature=0.1,
                system="You are a helpful AI that extracts political topics from text. Respond only with the most relevant topics, separated by commas.",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Extract the main political topics from this text that match the following categories: {', '.join(VALID_TOPICS)}
                        
                        Text: {chunk}
                        
                        Respond only with most relevant matching topics separated by commas. Ignore topics that are breifly mentioned. If one segment fits multiple topics, select the most relevant. If no topics are found, respond with 'NaN'."""
                    }
                ]
            )
            
            # Extract response
            result = message.content[0].text.strip()
            
            # Process topics
            chunk_topics = {topic.strip() for topic in result.split(',') 
                          if topic.strip() in VALID_TOPICS}
            all_topics.update(chunk_topics)
        
        return ', '.join(sorted(all_topics)) if all_topics else np.nan
    
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return np.nan

def process_transcripts(input_file: str, output_file: str) -> pd.DataFrame:
    """Process transcripts and extract topics."""
    try:
        # Load and clean data
        df = pd.read_csv(input_file)
        df = df[df['cleaned'].notna()].copy()
        
        # Extract topics
        print("Extracting topics...")
        df['claude_topics'] = df['cleaned'].apply(extract_topics)
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        return df
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    input_file = 'harris_topic_ryan.csv'
    output_file = 'harris_topics_claude_mini.csv'
    
    df = process_transcripts(input_file, output_file)
    if not df.empty:
        print("\nSample results:")
        print(df[['cleaned', 'Date', 'Title', 'URL', 'claude_topics']].head())