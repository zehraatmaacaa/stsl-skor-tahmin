import pandas as pd
import numpy as np
from scipy.stats import poisson

# We created the dataset manually with SQL
data = pd.read_csv("c:\\Users\ZEHRA\Downloads\stsl skor tahmin\SuperLig_Standings.csv")

# Average goals scored and conceded per game by each team
data["GoalsForPerMatch"] = data["GoalsFor"] / data["MatchesPlayed"]
data["GoalsAgainstPerMatch"] = data["GoalsAgainst"] / data["MatchesPlayed"]

# Score prediction with Simple Poisson Model
def predict_match(home_team, away_team, data):
    home_stats = data[data["TeamName"] == home_team].iloc[0]
    away_stats = data[data["TeamName"] == away_team].iloc[0]

    # Average goal odds
    home_attack = home_stats["GoalsForPerMatch"]
    home_defense = home_stats["GoalsAgainstPerMatch"]
    away_attack = away_stats["GoalsForPerMatch"]
    away_defense = away_stats["GoalsAgainstPerMatch"]

    # Home and away team goal expectation
    home_goals_exp = (home_attack + away_defense) / 2
    away_goals_exp = (away_attack + home_defense) / 2

    # Goal prediction with Poisson distribution
    home_goals = poisson.rvs(home_goals_exp)
    away_goals = poisson.rvs(away_goals_exp)

    return home_goals, away_goals

# Guess test
home_team = "Beşiktaş A.Ş."
away_team = "Fenerbahçe A.Ş."
predicted_home_goals, predicted_away_goals = predict_match(home_team, away_team, data)

print(f"Predicted Scor: {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}")

# Score prediction for all matches
match_predictions = []
for home_team in data["TeamName"]:
    for away_team in data["TeamName"]:
        if home_team != away_team:  
            home_goals, away_goals = predict_match(home_team, away_team, data)
            match_predictions.append({
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "HomeGoals": home_goals,
                "AwayGoals": away_goals
            })

# We save the predictions in a DataFrame
predictions_df = pd.DataFrame(match_predictions)

# We save the predictions in a CSV
predictions_df.to_csv("SuperLig_Predicted_Scores.csv", index=False)
print("Tüm maç tahminleri kaydedildi: SuperLig_Predicted_Scores.csv")