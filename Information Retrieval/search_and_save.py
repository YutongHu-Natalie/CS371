import json
import os

from openpyxl.styles.builtins import output

from sim_with_topic import SpeechSearcher


def get_next_file_id(output_dir: str) -> str:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    next_id = len(existing_files)
    return f"{next_id:03d}"


def run_searches(query: str, topics: list, output_dir: str):
    searcher = SpeechSearcher()
    file_id = get_next_file_id(output_dir)

    results = {
        "query_id": file_id,
        "query_text": query,
        "query_topics": topics,
        "results": {
            query: {
                "Trump": None,
                "Harris": None
            }
        }
    }

    for speaker in ["Trump", "Harris"]:
        search_results, scores = searcher.search(query, speaker=speaker, topics=topics, top_k=5)
        results["results"][query][speaker] = search_results

    output_path = os.path.join(output_dir, f"{file_id}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    query = input("Enter your search query: ")
    topics_input = input("Enter topics (comma-separated, or press Enter to skip): ")
    topics = [t.strip() for t in topics_input.split(",")] if topics_input.strip() else None

    output_dir = "Harris"
    run_searches(query, topics, output_dir)


if __name__ == "__main__":
    main()