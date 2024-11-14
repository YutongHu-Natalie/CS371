import torch
import torch.nn.functional as F
from conda.exports import root_dir
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv(dotenv_path='.env')

client = QdrantClient(url = os.getenv('QDRANT_URL'),
                      api_key= os.getenv('Qdrant_Key'))


revision = None  # Replace with the specific revision to ensure reproducibility if the model is updated.

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0", revision=revision)
model = model.to(device)

collection_name= "CS371"


def load_speeches(directory_path):
    speeches = []
    # Convert string path to Path object
    speech_dir = Path(directory_path)

    # Iterate through all JSON files in the directory
    for speech_file in sorted(speech_dir.glob('*.json')):
        print(speech_file)
        try:
            with open(speech_file, 'r', encoding='utf-8') as f:
                speech = json.load(f)
                speeches.append(speech)
        except Exception as e:
            print(f"Error reading {speech_file}: {e}")

    print(f"Loaded {len(speeches)} speeches from {directory_path}")
    return speeches



def upload_speeches_to_qdrant(speeches):
    for speech in speeches:
        # Combine all text segments into one list
        texts = [segment["text"] for segment in speech["content"]]
        # Generate embeddings for each text segment
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            device=device,
            batch_size=32,
            show_progress_bar=True
        )

        # Convert embeddings to numpy arrays
        embeddings_np = embeddings.cpu().numpy()

        # Upload each segment with metadata
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings_np)):
            points.append(PointStruct(
                id=f"{speech['speech_id']}_{i}",  # Unique ID for each segment
                vector=embedding.tolist(),
                payload={
                    "speaker": speech["speaker"],
                    "speech_id": speech["speech_id"],
                    "title": speech["title"],
                    "date": speech["date"],
                    "topics": speech["topics"],
                    "segment_id": i,
                    "text": text
                }
            ))

            client.upsert(
                collection_name=collection_name,
                points=points
            )

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))

    harris_speech_path= os.path.join(root_dir, "CS371/Content/Processed Materials/harris_speeches")
    trump_speech_path= os.path.join(root_dir, "CS371/Content/Processed Materials/trump_speeches")
    harris_speech= load_speeches(harris_speech_path)
    upload_speeches_to_qdrant(harris_speech)


if __name__ == "__main__":
    main()







