import json
import os
import pickle
import time
from typing import Dict, Tuple, List

from game import Game
from game_period import GamePeriod
from team import Team

DEFAULT_SAVE_LOCATION = './resources/predictions.pkl'
DEFAULT_STAT_LOCATION = './resources/predictions.json'


class Predictions:

    def __init__(self):
        self.historical_predictions: Dict[Game, List[Team]] = dict()
        self.instance_predictions: Dict[Game, List[Team]] = dict()

    def clear_instance_predictions(self):
        self.instance_predictions = dict()

    def add_prediction_instance(self, game: Game, prediction_prob: List[float]) -> Tuple:
        assert len(prediction_prob) == 2, f"There should only be 2 percentages in prediction_prob, found " \
                                          f"{len(prediction_prob)}: {prediction_prob}"
        self.historical_predictions.setdefault(game, list())
        self.instance_predictions.setdefault(game, list())
        actual_winner = game.home_team if game.home_team.scores.get(GamePeriod.TOTAL) > game.away_team.scores.get(
            GamePeriod.TOTAL) else game.away_team
        actual_loser = game.away_team if actual_winner is game.home_team else game.home_team
        predicted_winner = game.home_team if prediction_prob[0] > prediction_prob[1] else game.away_team
        predicted_loser = game.away_team if predicted_winner is game.home_team else game.home_team
        print(f"PREDICTED: {predicted_winner.name} beats {predicted_loser.name} for game {game.code}")
        print(f"ACTUAL RESULT: {actual_winner.name} beats {actual_loser.name} for game {game.code}\n")
        self.historical_predictions[game].append(predicted_winner)
        self.instance_predictions[game].append(predicted_winner)
        actual_home_team_won = True if actual_winner == game.home_team else False
        predicted_home_team_won = True if predicted_winner == game.home_team else False
        return actual_home_team_won, predicted_home_team_won

    def add_seasonal_prediction_instance(self, predictions: List, sorted_games: List[Game]):
        confusion_matrix = [0, 0, 0, 0]
        for i, game in enumerate(sorted_games):
            probabilities = predictions[i]['probabilities']
            outcomes = self.add_prediction_instance(game, probabilities)
            confusion_matrix = recursive_build_confusion_matrix(confusion_matrix, outcomes[0], outcomes[1])
        for k, v in analyze_confusion_matrix(confusion_matrix, len(sorted_games)).items():
            print(f"{k} : {v}")

    def analyze_end_performance(self):
        print(f"###### FINAL RESULTS FOR EACH GAME###### \n")
        total_results = [0, 0, 0, 0]
        for game in self.instance_predictions:
            print(f"Simulated game {game.home_team.name} vs {game.away_team.name} "
                  f"{len(self.instance_predictions[game])} times.")
            confusion_matrix = [0, 0, 0, 0]
            for predicted_winner in self.instance_predictions[game]:
                actual_winner = game.home_team if game.home_team.scores.get(
                    GamePeriod.TOTAL) > game.away_team.scores.get(GamePeriod.TOTAL) else game.away_team
                actual_home_team_won = True if actual_winner == game.home_team else False
                predicted_home_team_won = True if predicted_winner == game.home_team else False
                confusion_matrix = recursive_build_confusion_matrix(confusion_matrix, actual_home_team_won,
                                                                    predicted_home_team_won)
            total_results[0] += confusion_matrix[0]
            total_results[1] += confusion_matrix[1]
            total_results[2] += confusion_matrix[2]
            total_results[3] += confusion_matrix[3]

            for k, v in analyze_confusion_matrix(confusion_matrix, len(self.instance_predictions[game])).items():
                print(f"{k} : {v}")
        print("\n###### FINAL RESULTS FOR ALL GAMES###### \n")
        final_stats = analyze_confusion_matrix(total_results, sum(total_results))
        for k, v in final_stats.items():
            print(f"{k} : {v}")
        self.write_stats_to_json(final_stats)
        self.clear_instance_predictions()

    def save_predictions_instance(self, path: str = DEFAULT_SAVE_LOCATION):
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def write_stats_to_json(self, stats: Dict, path: str = DEFAULT_STAT_LOCATION):
        if os.path.isfile(path):
            with open(path, 'a') as json_file:
                json.dump({time.time(): stats}, json_file)
        else:
            with open(path, 'w') as json_file:
                json.dump({time.time(): stats}, json_file)


def recursive_build_confusion_matrix(previous: List[int], actual_home_win: bool, predicted_home_win: bool) -> List[int]:
    """
 [ True Positive, False Negative, False Positive, True Negative]
    """
    if predicted_home_win and actual_home_win:
        previous[0] += 1
    elif not predicted_home_win and actual_home_win:
        previous[1] += 1
    elif predicted_home_win and not actual_home_win:
        previous[2] += 1
    elif not predicted_home_win and not actual_home_win:
        previous[3] += 1
    return previous


def analyze_confusion_matrix(confusion_matrix: List[int], num_games: int) -> Dict[str, float]:
    true_positives = confusion_matrix[0]
    false_negatives = confusion_matrix[1]
    false_positives = confusion_matrix[2]
    true_negatives = confusion_matrix[3]
    accuracy = (true_positives + true_negatives) / num_games
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0
    error_rate = (false_negatives + false_positives) / num_games
    return {"True Positives": true_positives, "False Negatives": false_negatives, "False Positives": false_positives,
            "True Negatives": true_negatives, "Accuracy": accuracy, "Precision": precision, "Error Rate": error_rate}


def load_predictions(path: str = DEFAULT_SAVE_LOCATION) -> Predictions:
    if not os.path.isfile(path):
        return Predictions()
    else:
        with open(path, 'rb') as pickle_file:
            predictions: Predictions = pickle.load(pickle_file)
        return predictions