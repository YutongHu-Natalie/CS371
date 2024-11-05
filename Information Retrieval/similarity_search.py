import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from typing import Dict, List
import sqlite3


class SpeechEmbedding:
    def __init__(self, model_name="avsolatorio/GIST-large-Embedding-v0"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.conn = None
        nltk.download('punkt')

    def split_into_sentences(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)

    def process_and_store(self, speeches: Dict[str, str], db_path: str = 'speeches.db'):
        # Connect to SQLite
        self.conn = sqlite3.connect(db_path)
        c = self.conn.cursor()

        # Create table
        c.execute('''CREATE TABLE IF NOT EXISTS speeches
                    (id INTEGER PRIMARY KEY,
                     sentence TEXT,
                     candidate TEXT)''')

        # Process all speeches
        all_embeddings = []
        idx = 0

        for candidate, speech in speeches.items():
            # Split into sentences
            sentences = self.split_into_sentences(speech)

            # Process in batches
            batch_size = 32
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]

                # Create embeddings
                embeddings = self.model.encode(batch)
                all_embeddings.extend(embeddings)

                # Store sentences in SQLite
                for sentence in batch:
                    c.execute("INSERT INTO speeches VALUES (?, ?, ?)",
                              (idx, sentence, candidate))
                    idx += 1

        # Convert to numpy array
        all_embeddings = np.array(all_embeddings)

        # Initialize FAISS index
        dimension = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Add vectors to the index
        self.index.add(all_embeddings)

        # Save changes and close connection
        self.conn.commit()

        # Save FAISS index
        faiss.write_index(self.index, "speeches.index")

        print(f"Processed and stored {idx} sentences")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        # Encode query
        query_vector = self.model.encode([query])

        # Search in FAISS
        scores, indices = self.index.search(query_vector, k)

        # Get corresponding sentences from SQLite
        results = []
        c = self.conn.cursor()

        for idx, score in zip(indices[0], scores[0]):
            c.execute("SELECT sentence, candidate FROM speeches WHERE id=?", (int(idx),))
            sentence, candidate = c.fetchone()
            results.append({
                'sentence': sentence,
                'candidate': candidate,
                'score': float(score)
            })

        return results

    def load_index(self, index_path: str = "speeches.index", db_path: str = "speeches.db"):
        self.index = faiss.read_index(index_path)
        self.conn = sqlite3.connect(db_path)


# Usage example:
if __name__ == "__main__":
    # Initialize
    embedder = SpeechEmbedding()

    # Example speeches
    speeches = {
        "candidate_1": """First speech content here. This is about economic policy.
                         We need to focus on growth. Jobs are important.""",
        "candidate_2": """Second speech content here. The economy needs reform.
                         We should reduce taxes. Employment is our priority."""
    }

    # Process and store speeches
    embedder.process_and_store(speeches)

    # Example debate question
    debate_question = "What is your stance on economic policy?"

    # Search for relevant sentences
    results = embedder.search(debate_question)

    # Print results
    print("\nResults for:", debate_question)
    for result in results:
        print(f"\nCandidate: {result['candidate']}")
        print(f"Statement: {result['sentence']}")
        print(f"Similarity Score: {result['score']:.3f}")