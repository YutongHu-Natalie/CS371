'''
It intends to segment the speech transcripts while keeping their important information.
It will be implemented on "{candidate}_transcripts_with_topics.csv"
'''

import pandas as pd
from datetime import datetime
import re
import os


class TranscriptProcessor:
    def __init__(self, output_dir="processed_speeches", min_segment_length=50, max_segment_length=512):
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.output_dir = output_dir
        self.speech_counter = 0

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_speech_id(self):
        """Generate a 3-digit speech ID"""
        speech_id = f"{self.speech_counter:03d}"
        self.speech_counter += 1
        return speech_id

    def standardize_date(self, date_str):
        """Convert date string to YYYYMMDD format"""
        date_obj = datetime.strptime(date_str, '%b %d, %Y')
        return date_obj.strftime('%Y%m%d')

    def segment_text(self, text):
        """Split text into segments while trying to maintain sentence boundaries"""

        #clean time
        text= re.sub(r'(\d{2}:\d{2})', "", text)

        # transform unicode
        replacements = {
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--',  # Em dash
            '\u2026': '...',  # Horizontal ellipsis
        }

        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)

        #split text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        segments = []
        current_segment = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.max_segment_length and current_segment:
                segments.append(' '.join(current_segment))
                current_segment = []
                current_length = 0

            current_segment.append(sentence)
            current_length += sentence_length

            if current_length >= self.max_segment_length:
                segments.append(' '.join(current_segment))
                current_segment = []
                current_length = 0

        if current_segment:
            segments.append(' '.join(current_segment))

        return segments

    def process_transcript(self, title, date, url, topics, text):
        """Process a single transcript into segments with metadata"""
        speech_id = self.get_speech_id()
        date_standard = self.standardize_date(date)
        segments = self.segment_text(text)

        processed_segments = []
        for i, segment in enumerate(segments):
            if len(segment.split()) < self.min_segment_length:
                continue

            processed_segments.append({
                'text': segment
            })

        # Save speech metadata
        speech = {
            'speech_id': speech_id,
            'title': title,
            'date': date_standard,
            'url': url,
            'topics': topics,
            'num_segments': len(processed_segments),
            'content': processed_segments

        }

        # Save metadata to file
        metadata_path = os.path.join(self.output_dir, f"{speech_id}.json")
        with open(metadata_path, 'w') as f:
            import json
            json.dump(speech, f, indent=2)

        return processed_segments

    def process_multiple_transcripts(self, transcripts_df):
        """Process multiple transcripts from a DataFrame"""
        all_segments = []
        speech_metadata = {}

        # Sort by date to ensure consistent ID assignment
        transcripts_df = transcripts_df.sort_values('Date')

        for _, row in transcripts_df.iterrows():
            segments = self.process_transcript(
                title=row['Title'],
                date=row['Date'],
                url=row['URL'],
                topics=row['topic'],
                text=row['cleaned']
            )




# Example usage:
candidate= "harris"

# Go up two levels from current script location to reach Contenet directory
script_dir = os.path.dirname(os.path.abspath(__file__))
contenet_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to root directory

# Construct paths relative to Contenet directory
input_path = os.path.join(contenet_dir, "Content", "Raw Materials", f"{candidate}_transcripts_with_topics (v2).csv")
output_dir = os.path.join(contenet_dir, "Content", "Processed Materials", f"{candidate}_speeches")

print(contenet_dir)
# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the CSV and process
df = pd.read_csv(input_path)
processor = TranscriptProcessor(output_dir=output_dir)
processed_df = processor.process_multiple_transcripts(df)


