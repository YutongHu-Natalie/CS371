import os
import sys
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.http.models import VectorParams, PointStruct, Filter
from dotenv import load_dotenv

def initialize_qdrant_client():
    """Initialize Qdrant client with error handling and connection validation."""
    load_dotenv()
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_key = os.getenv('QDRANT_KEY')

    if not qdrant_url:
        raise ValueError("QDRANT_URL not found in environment variables")
    if not qdrant_key:
        raise ValueError("QDRANT_KEY not found in environment variables")

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        # Test connection
        client.get_collections()
        return client
    except Exception as e:
        print(f"Failed to connect to Qdrant server: {e}")
        sys.exit(1)

def initialize_model(model_name="all-MiniLM-L6-v2"):
    """Initialize the sentence transformer model."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = SentenceTransformer(model_name)
        return model.to(device), device
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

def search_similar(client, collection_name, query_embedding, limit=5):
    """Search for similar sentences in Qdrant."""
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit,
        with_payload=True  # Include metadata in results
    )
    
    print(f"Top {limit} similar results:")
    for result in search_results:
        print(f"ID: {result.id}, Score: {result.score}, Text: {result.payload['text']}")
    return search_results

def fetch_context(client, collection_name, transcript_id, segment_id, context_range=(-1, 1)):
    """Fetch context sentences around a specific segment."""
    context_sentences = []
    for offset in range(context_range[0], context_range[1] + 1):
        target_segment_id = segment_id + offset
        results = client.search(
            collection_name=collection_name,
            query_filter={
                "must": [
                    {"key": "speech_id", "match": {"value": transcript_id}},
                    {"key": "segment_id", "match": {"value": target_segment_id}}
                ]
            },
            limit=1,
            with_payload=True
        )
        if results:
            context_sentences.append(results[0].payload["text"])
    return context_sentences

def search_with_context(client, model, query_text, device, collection_name, context_range=(-1, 1), limit=5):
    """Perform similarity search and fetch contextual sentences."""
    # Generate embedding for the query
    query_embedding = model.encode(query_text, convert_to_tensor=True, device=device).cpu().numpy()
    
    # Search for similar sentences
    search_results = search_similar(client, collection_name, query_embedding.tolist(), limit)
    
    # Fetch context for the top result
    if search_results:
        top_result = search_results[0]
        transcript_id = top_result.payload["speech_id"]
        segment_id = top_result.payload["segment_id"]
        context = fetch_context(client, collection_name, transcript_id, segment_id, context_range)
        
        print("\nQuery Text:", query_text)
        print("Top Match:", top_result.payload["text"])
        print("Context Sentences:", context)
    else:
        print("No similar sentences found.")

def main():
    try:
        # Initialize Qdrant client and model
        client = initialize_qdrant_client()
        model, device = initialize_model()

        # Define collection and query
        collection_name = "CS371"
        query_text = "We must work together to achieve peace and prosperity." # modify this for different sentences

        # Perform similarity search with context retrieval
        search_with_context(client, model, query_text, device, collection_name)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
