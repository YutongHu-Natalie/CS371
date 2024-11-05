import spacy
import json
import re
from typing import List, Dict
from tqdm import tqdm  # For progress bars


class DebateProcessor:
    def __init__(self):
        # Load spaCy model with specific pipeline components we need
        self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        # Custom sentence boundary rules for debate text
        self.nlp.add_pipe("sentencizer")

    def preprocess_text(self, text: str) -> str:
        """Clean and standardize the debate text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Standardize speaker markers
        text = re.sub(r'\*\*([^:]+):\*\*', r'\n**\1:**', text)
        return text.strip()

    def extract_turns(self, text: str) -> List[Dict]:
        """Extract speaking turns from debate text"""
        # Split text into turns based on speaker markers
        turns = []
        segments = text.split('\n**')

        for segment in segments:
            if ':' not in segment:
                continue

            speaker, content = segment.split(':**', 1)
            turns.append({
                'speaker': speaker.strip(),
                'content': content.strip()
            })

        return turns

    def segment_into_statements(self, turns: List[Dict]) -> List[Dict]:
        """Convert turns into individual statements with annotation structure"""
        statements = []
        statement_id = 1

        for turn in tqdm(turns, desc="Processing statements"):
            # Process content into sentences
            doc = self.nlp(turn['content'])

            for sent in doc.sents:
                text = sent.text.strip()
                if not text:  # Skip empty statements
                    continue

                statement = {
                    "statement_id": f"S{statement_id}",
                    "speaker": turn['speaker'],
                    "text": text,
                    "annotator_id": "",  # To be filled by annotator
                    "verifiable":None  # To be filled by annotator; 0-> unverifiable, 1-> verifiable by fact checking, 2-> verifiable by contradiction detection
                }
                statements.append(statement)
                statement_id += 1

        return statements


def process_debate_file(input_file: str, output_file: str):
    """Process entire debate file and save statements"""
    processor = DebateProcessor()

    # Read input file
    print("Reading debate file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Process text
    print("Preprocessing text...")
    text = processor.preprocess_text(text)

    print("Extracting speaking turns...")
    turns = processor.extract_turns(text)

    print("Segmenting into statements...")
    statements = processor.segment_into_statements(turns)

    # Save to JSON file
    print(f"Saving {len(statements)} statements...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(statements, f, indent=2, ensure_ascii=False)

    print("Processing complete!")
    return statements

'''
# Function to split statements into manageable chunks for annotation
def create_annotation_batches(statements: List[Dict], batch_size: int = 100):
    """Split statements into smaller files for easier annotation"""
    total_batches = (len(statements) + batch_size - 1) // batch_size

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(statements))
        batch = statements[start_idx:end_idx]

        # Save batch
        batch_file = f"batch_{i + 1}_statements.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        print(f"Created {batch_file} with {len(batch)} statements")

'''


# Usage example:
if __name__ == "__main__":
    # Process the entire debate
    statements = process_debate_file(
        input_file="debate_transcript.txt",
        output_file="all_statements.json"
    )

    # Create smaller batches for annotation
    #create_annotation_batches(statements, batch_size=100)