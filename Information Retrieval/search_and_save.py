import json
import os
from sim_with_topic import SpeechSearcher


def get_next_file_id(output_dir: str) -> str:
    """Generate the next available file ID in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    next_id = len(existing_files)
    return f"{next_id:03d}"


def run_search(query: str, speaker: str, topics: list, output_dir: str):
    """Run search for a single speaker and save results."""
    searcher = SpeechSearcher()
    file_id = get_next_file_id(output_dir)

    results = {
        "query_id": file_id,
        "query_text": query,
        "query_topics": topics,
        "speaker": speaker,
        "results": {}
    }

    search_results, scores = searcher.search(query, speaker=speaker, topics=topics, top_k=3)
    results["results"][query] = search_results
    print(f"found {len(search_results)} matched statements")

    output_path = os.path.join(output_dir, f"{file_id}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    # Get the speaker selection
    print("\nAvailable speakers:")
    print("1. Trump")
    print("2. Harris")

    while True:
        try:
            #speaker_choice = int(input("\nSelect speaker (1 or 2): "))
            speaker_choice= 1
            if speaker_choice in [1, 2]:
                selected_speaker = "Trump" if speaker_choice == 1 else "Harris"
                break
            print("Invalid selection. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get search parameters
    query = input("Enter your search query: ")
    topics_input = input("Enter topics (comma-separated, or press Enter to skip): ")
    topics = [t.strip() for t in topics_input.split(",")] if topics_input.strip() else None

    # Create output directory based on selected speaker
    output_dir = selected_speaker
    run_search(query, selected_speaker, topics, output_dir)


if __name__ == "__main__":
    main()