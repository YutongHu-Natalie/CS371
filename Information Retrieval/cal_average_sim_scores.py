import json
import os
from statistics import mean


def calculate_and_update_scores(directory: str):
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        for query, speakers in data['results'].items():
            speaker_scores = {}
            for speaker in ['Trump', 'Harris']:
                results = speakers.get(speaker, [])
                if results:
                    scores = [result['match_score'] for result in results]
                    avg_score = round(mean(scores) if scores else 0, 4)
                    speaker_scores[f"{speaker}_avg_score"] = avg_score

            speakers.update(speaker_scores)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Updated {filename} with average scores")


if __name__ == "__main__":
    calculate_and_update_scores("Trump")