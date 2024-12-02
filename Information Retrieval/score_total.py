import json
import os
from statistics import mean
from collections import defaultdict


def calculate_and_update_scores(speaker):
    speaker_scores = []
    num_fetch= 0
    for filename in os.listdir(speaker):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(speaker, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        for fetched_stm in data['results'].values():
            speaker_scores.append(fetched_stm[0].get("match_score"))
    avg_score = mean(speaker_scores)

    return avg_score





if __name__ == "__main__":
    print(calculate_and_update_scores("Trump"))