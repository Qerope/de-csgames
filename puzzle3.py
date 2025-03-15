import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

with open("10k_games.json", "r") as file:
    games_data = json.load(file)

def extract_features(game):
    card_suits = [card["suit"] for card in game["cards"]]
    card_ranks = [card["rank"] for card in game["cards"]]
    suit_counts = dict(Counter(card_suits))
    rank_counts = dict(Counter(card_ranks))
    
    return {
        "gameId": game["id"],
        "isPlayed": game["isPlayed"],
        "num_hearts": suit_counts.get("HEARTS", 0),
        "num_diamonds": suit_counts.get("DIAMONDS", 0),
        "num_clubs": suit_counts.get("CLUBS", 0),
        "num_spades": suit_counts.get("SPADES", 0),
        "num_pairs": sum(1 for count in rank_counts.values() if count == 2),
        "num_triples": sum(1 for count in rank_counts.values() if count == 3),
        "num_quads": sum(1 for count in rank_counts.values() if count == 4),
        "highest_card": max(card_ranks),
        "lowest_card": min(card_ranks),
        "actions": game["actions"],
        "outcome": game["outcome"]
    }

df = pd.DataFrame([extract_features(game) for game in games_data])

action_columns = [f"action_{i+1}" for i in range(15)]
df[action_columns] = pd.DataFrame(df["actions"].tolist(), index=df.index)
df.drop(columns=["actions"], inplace=True)

correlation_matrix = df.corr()["outcome"].sort_values(ascending=False)
print("\nTop correlated actions/features with outcome:")
print(correlation_matrix)

plt.figure(figsize=(10, 5))
plt.hist(df["outcome"], bins=20, edgecolor='black')
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.title("Distribution of Game Outcomes")
plt.show()

def most_common_action_combos(df, top_n=10):
    action_sequences = [tuple(actions) for actions in df[df["outcome"] > 0][action_columns].values]
    common_combos = Counter(action_sequences).most_common(top_n)
    return common_combos

print("\nMost common action sequences for successful outcomes:")
print(most_common_action_combos(df))
