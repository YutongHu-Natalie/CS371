import json
import os
from statistics import mean
from collections import defaultdict


def calculate_and_update_scores(directory):
    all_trump_scores = []
    all_harris_scores = []
    all_combined_scores = []

    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        for query, speakers in data['results'].items():
            speaker_scores = {}
            trump_results = speakers.get('Trump', [])
            harris_results = speakers.get('Harris', [])

            # Calculate Trump average
            if trump_results:
                trump_scores = [result['match_score'] for result in trump_results]
                trump_avg = round(mean(trump_scores), 4)
                speaker_scores['Trump_avg_score'] = trump_avg
                all_trump_scores.append(trump_avg)

            # Calculate Harris average
            if harris_results:
                harris_scores = [result['match_score'] for result in harris_results]
                harris_avg = round(mean(harris_scores), 4)
                speaker_scores['Harris_avg_score'] = harris_avg
                all_harris_scores.append(harris_avg)

            # Calculate combined average
            combined_scores = []
            if trump_results:
                combined_scores.extend(trump_scores)
            if harris_results:
                combined_scores.extend(harris_scores)
            if combined_scores:
                combined_avg = round(mean(combined_scores), 4)
                speaker_scores['Combined_avg_score'] = combined_avg
                all_combined_scores.append(combined_avg)

            speakers.update(speaker_scores)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    # Calculate overall averages across all files
    overall_stats = {
        'Overall_Trump_avg': round(mean(all_trump_scores), 4) if all_trump_scores else 0,
        'Overall_Harris_avg': round(mean(all_harris_scores), 4) if all_harris_scores else 0,
        'Overall_Combined_avg': round(mean(all_combined_scores), 4) if all_combined_scores else 0
    }

    # Save overall statistics
    with open(os.path.join(directory, 'overall_statistics.json'), 'w') as f:
        json.dump(overall_stats, f, indent=2)

    print(f"Updated all files with average scores")
    print("Overall statistics:", overall_stats)


if __name__ == "__main__":
    calculate_and_update_scores("Harris")