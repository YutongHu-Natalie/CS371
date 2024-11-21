import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest
from typing import List, Dict, Any
import os
from dotenv import load_dotenv


class SpeechSearcher:
    def __init__(self, model_name: str = "avsolatorio/GIST-large-Embedding-v0"):
        """Initialize the searcher with model and Qdrant client."""
        self.model = SentenceTransformer(model_name)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Qdrant client
        load_dotenv()
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_key = os.getenv('Qdrant_Key')
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    def _get_context_segments(self, speech_id: str, segment_id: int, window: int = 1) -> List[Dict[str, Any]]:
        """Retrieve context segments before and after the given segment."""
        context_points = []

        # Calculate the range of segment IDs to fetch
        start_id = generate_point_id(speech_id, max(0, segment_id - window))
        end_id = generate_point_id(speech_id, segment_id + window)

        # Fetch all points in the range
        points = self.client.retrieve(
            collection_name="CS371",
            ids=list(range(start_id, end_id + 1))
        )

        return [point.payload for point in points]

    def search(self, query: str, top_k: int = 5, context_window: int = 1) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return results with context.

        Args:
            query: Search query text
            top_k: Number of top results to return
            context_window: Number of segments to include before and after each match

        Returns:
            List of dictionaries containing search results with context
        """
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy()

            # Perform the search
            search_results = self.client.search(
                collection_name="CS371",
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=0.7  # Adjust this threshold as needed
            )

            # Process results and add context
            detailed_results = []
            for result in search_results:
                payload = result.payload
                score = result.score

                # Get context segments
                context_segments = self._get_context_segments(
                    payload["speech_id"],
                    payload["segment_id"],
                    context_window
                )

                # Create detailed result entry
                detailed_result = {
                    "match_score": score,
                    "metadata": {
                        "speaker": payload["speaker"],
                        "speech_id": payload["speech_id"],
                        "title": payload["title"],
                        "date": payload["date"],
                        "topics": payload["topics"]
                    },
                    "matched_segment": {
                        "segment_id": payload["segment_id"],
                        "text": payload["text"]
                    },
                    "context": context_segments
                }
                detailed_results.append(detailed_result)

            return detailed_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []


def generate_point_id(speech_id: str, segment_index: int) -> int:
    """Generate a unique integer ID from speech ID and segment index."""
    speech_num = int(speech_id)
    return (speech_num * 1000) + segment_index


# Example usage
def main():
    # Initialize the searcher
    searcher = SpeechSearcher()

    # Example search query
    query = "And I intend on extending a tax cut for those families of $6,000, which is the largest child tax credit that we have given in a long time."
    results = searcher.search(query, top_k=5, context_window=1)

    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['match_score']:.3f})")
        print("\nMetadata:")
        for key, value in result["metadata"].items():
            print(f"{key}: {value}")

        print("\nMatched Segment:")
        print(result["matched_segment"]["text"])

        print("\nContext:")
        for ctx in result["context"]:
            print(f"[Segment {ctx['segment_id']}]: {ctx['text']}")
        print("-" * 80)


if __name__ == "__main__":
    main()