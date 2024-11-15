import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException
import sys

def initialize_qdrant_client():
    """Initialize Qdrant client with error handling and connection validation."""
    load_dotenv()

    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_key = os.getenv('Qdrant_Key')

    if not qdrant_url:
        raise ValueError("QDRANT_URL not found in environment variables")
    if not qdrant_key:
        raise ValueError("Qdrant_Key not found in environment variables")

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        # Test connection
        client.get_collections()
        return client
    except ResponseHandlingException as e:
        print(f"Failed to connect to Qdrant server: {e}")
        print("Please check if:")
        print("1. The Qdrant server is running")
        print("2. The URL and API key are correct")
        print("3. There are no network/firewall issues")
        sys.exit(1)

def generate_point_id(speech_id, segment_index):
    """Generate a unique integer ID from speech ID and segment index."""
    # Convert speech_id (e.g., "000") to integer and combine with segment_index
    speech_num = int(speech_id)
    # Multiply speech_num by 1000 to leave room for up to 999 segments per speech
    return (speech_num * 1000) + segment_index

def upload_speeches_to_qdrant(client, model, device, speeches, collection_name, batch_size=32):
    """Upload speeches to Qdrant with batch processing and error handling."""
    try:
        # Ensure collection exists with correct parameters
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)

        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
            )

        for speech in speeches:
            try:
                texts = [segment["text"] for segment in speech["content"]]

                # Generate embeddings with batching
                embeddings = model.encode(
                    texts,
                    convert_to_tensor=True,
                    device=device,
                    batch_size=batch_size,
                    show_progress_bar=True
                )

                embeddings_np = embeddings.cpu().numpy()

                # Create points in batches
                batch_size = 100  # Adjust based on your needs
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = embeddings_np[i:i + batch_size]

                    points = [
                        PointStruct(
                            id=generate_point_id(speech["speech_id"], j),
                            vector=embedding.tolist(),
                            payload={
                                "speaker": speech["speaker"],
                                "speech_id": speech["speech_id"],
                                "title": speech["title"],
                                "date": speech["date"],
                                "topics": speech["topics"],
                                "segment_id": j,
                                "text": text
                            }
                        )
                        for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings), start=i)
                    ]

                    client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    print(f"Uploaded batch {i // batch_size + 1} for speech {speech['speech_id']}")

            except Exception as e:
                print(f"Error processing speech {speech['speech_id']}: {e}")
                continue

    except Exception as e:
        print(f"Error uploading to Qdrant: {e}")
        raise

def initialize_model(model_name="avsolatorio/GIST-large-Embedding-v0", revision=None):
    """Initialize the sentence transformer model with proper error handling."""
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        model = SentenceTransformer(model_name, revision=revision)
        return model.to(device), device
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

def load_speeches(directory_path):
    """Load speeches with enhanced error handling and validation."""
    speeches = []
    speech_dir = Path(directory_path)

    if not speech_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    files = sorted(speech_dir.glob('*.json'))
    if not files:
        print(f"Warning: No JSON files found in {directory_path}")
        return speeches

    for speech_file in files:
        try:
            with open(speech_file, 'r', encoding='utf-8') as f:
                speech = json.load(f)
                # Validate required fields
                required_fields = ['speech_id', 'speaker', 'title', 'date', 'topics', 'content']
                missing_fields = [field for field in required_fields if field not in speech]
                if missing_fields:
                    print(f"Warning: Skipping {speech_file}. Missing required fields: {missing_fields}")
                    continue
                speeches.append(speech)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {speech_file}: {e}")
        except Exception as e:
            print(f"Error reading {speech_file}: {e}")

    print(f"Loaded {len(speeches)} speeches from {directory_path}")
    return speeches

def main():
    try:
        # Initialize clients and models
        client = initialize_qdrant_client()
        model, device = initialize_model()

        # Set up paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(script_dir))

        harris_speech_path = os.path.join(root_dir, "CS371/Content/Processed Materials/harris_speeches")
        trump_speech_path = os.path.join(root_dir, "CS371/Content/Processed Materials/trump_speeches")

        # Load and process speeches
        harris_speeches = load_speeches(harris_speech_path)
        if harris_speeches:
            upload_speeches_to_qdrant(client, model, device, harris_speeches, "CS371")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()