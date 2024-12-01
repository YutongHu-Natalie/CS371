import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional, Tuple
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

        # Available speakers
        self.speakers = ["Trump", "Harris"]

    def _get_context_segment(self, speech_id: str, segment_id: int, direction: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single context segment either before or after the given segment.

        Args:
            speech_id: ID of the speech
            segment_id: Current segment ID
            direction: 1 for next segment, -1 for previous segment
        """
        try:
            target_id = generate_point_id(speech_id, segment_id + direction)
            points = self.client.retrieve(
                collection_name="CS371",
                ids=[target_id]
            )
            return points[0].payload if points else None
        except Exception:
            return None

    def search(self, query: str, speaker: Optional[str] = None, topics: Optional[List[str]] = None, top_k: int = 3) -> \
    Tuple[
        List[Dict[str, Any]], List[float]]:
        """
        Perform similarity search and return results with immediate context.
        """
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy()

            # Create filters if specified
            filters = []
            if speaker:
                filters.append(FieldCondition(key="speaker", match=MatchValue(value=speaker)))
            if topics:
                filters.append(FieldCondition(key="topics", match=MatchValue(value=topic)) for topic in topics)

            search_filter = Filter(must=filters) if filters else None

            # First, get all results without threshold to analyze scores
            all_results = self.client.search(
                collection_name="CS371",
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=0.0,
                query_filter=search_filter
            )

            all_scores = [result.score for result in all_results]

            # Filter results with high similarity and respect top_k limit
            high_similarity_results = [result for result in all_results if result.score >= 0.5][:top_k]

            # Process results and add context
            detailed_results = []
            for result in high_similarity_results:
                payload = result.payload
                score = result.score
                segment_id = payload["segment_id"]

                # Get previous and next segments
                prev_segment = self._get_context_segment(payload["speech_id"], segment_id, -1)
                next_segment = self._get_context_segment(payload["speech_id"], segment_id, 1)

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
                        "segment_id": segment_id,
                        "text": payload["text"]
                    },
                    "context": {
                        "previous": prev_segment["text"] if prev_segment else None,
                        "next": next_segment["text"] if next_segment else None
                    }
                }
                detailed_results.append(detailed_result)

            return detailed_results, all_scores

        except Exception as e:
            print(f"Error during search: {e}")
            return [], []

    def get_available_speakers(self) -> List[str]:
        """Return list of available speakers."""
        return self.speakers


def generate_point_id(speech_id: str, segment_index: int) -> int:
    """Generate a unique integer ID from speech ID and segment index."""
    speech_num = int(speech_id)
    return (speech_num * 1000) + segment_index


def main():
    # Initialize the searcher
    searcher = SpeechSearcher()

    # Get available speakers
    speakers = searcher.get_available_speakers()

    while True:
        # Speaker selection
        print("\nAvailable speakers:")
        print("0. All speakers")
        for i, speaker in enumerate(speakers, 1):
            print(f"{i}. {speaker}")

        try:
            speaker_choice = input("\nSelect speaker number (or 'quit' to exit): ")
            if speaker_choice.lower() == 'quit':
                break

            speaker_idx = int(speaker_choice)
            selected_speaker = None if speaker_idx == 0 else speakers[speaker_idx - 1]

            if selected_speaker:
                print(f"\nSearching in {selected_speaker}'s speeches")
            else:
                print("\nSearching in all speeches")

        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
            continue

        # Topics filter
        topic_filter = input("\nEnter topics to filter by (comma-separated, or press Enter to skip): ")
        topics = [topic.strip() for topic in topic_filter.split(",")] if topic_filter else None

        # Get search query
        query = input("\nEnter your search query: ")

        results, all_scores = searcher.search(query, speaker=selected_speaker, topics=topics, top_k=3)

        # Provide feedback based on search results
        if not results:
            print("\nNo high-similarity matches found.")
            if all_scores:
                print(f"Best match score was: {max(all_scores):.3f} (threshold is 0.5)")
                print("Try:")
                print("1. Using different key phrases")
                print("2. Searching for shorter, specific phrases")
                print("3. Making sure the content exists in the speeches")
            continue

        # Display results
        print(f"\nFound {len(results)} high-similarity matches:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['match_score']:.3f})")
            print("\nMetadata:")
            for key, value in result["metadata"].items():
                print(f"{key}: {value}")

            print("\nContext:")
            if result["context"]["previous"]:
                print("\nPrevious segment:")
                print(result["context"]["previous"])

            print("\nMatched Segment:")
            print(result["matched_segment"]["text"])

            if result["context"]["next"]:
                print("\nNext segment:")
                print(result["context"]["next"])

            print("-" * 80)


if __name__ == "__main__":
    main()
