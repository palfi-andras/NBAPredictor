import datetime
import json
import os
import logging
from typing import Dict, Tuple, List

import tensorflow as tf

from game import Game
from game_period import GamePeriod
from team import Team


class Predictions:

    def __init__(self, season: str, num_epochs: int, nn_shape: List[int], labels_used: List[str], outfile: str,
            logger: logging):
        self.season = season
        self.outfile = outfile
        self.instance_predictions: Dict[Game, List[Team]] = dict()
        self.labels_used = labels_used
        self.best_accuracy = float('-inf')
        self.best_vars = None
        self.num_epochs = num_epochs
        self.nn_shape = nn_shape
        self.logger = logger

    def add_prediction_instance(self, game: Game, prediction_prob: List[float]) -> Tuple:
        assert len(prediction_prob) == 2, f"There should only be 2 percentages in prediction_prob, found " \
                                          f"{len(prediction_prob)}: {prediction_prob}"
        self.instance_predictions.setdefault(game, list())
        actual_winner = game.home_team if game.home_team.scores.get(GamePeriod.TOTAL) > game.away_team.scores.get(
            GamePeriod.TOTAL) else game.away_team
        actual_loser = game.away_team if actual_winner is game.home_team else game.home_team
        predicted_winner = game.home_team if prediction_prob[0] > prediction_prob[1] else game.away_team
        predicted_loser = game.away_team if predicted_winner is game.home_team else game.home_team
        self.logger.info(f"PREDICTED: {predicted_winner.name} beats {predicted_loser.name} for game {game.code}")
        self.logger.info(f"ACTUAL RESULT: {actual_winner.name} beats {actual_loser.name} for game {game.code}\n")
        self.instance_predictions[game].append(predicted_winner)
        actual_home_team_won = True if actual_winner == game.home_team else False
        predicted_home_team_won = True if predicted_winner == game.home_team else False
        return actual_home_team_won, predicted_home_team_won

    def add_seasonal_prediction_instance(self, predictions: List, sorted_games: List[Game],
            tf_model: tf.estimator.DNNClassifier):
        confusion_matrix = [0, 0, 0, 0]
        for i, game in enumerate(sorted_games):
            probabilities = predictions[i]['probabilities']
            outcomes = self.add_prediction_instance(game, probabilities)
            confusion_matrix = recursive_build_confusion_matrix(confusion_matrix, outcomes[0], outcomes[1])
        for k, v in analyze_confusion_matrix(confusion_matrix, len(sorted_games)).items():
            if k == "Accuracy" and v > self.best_accuracy:
                self.best_accuracy = v
                self.best_vars = {name: tf_model.get_variable_value(name).tolist() for name in
                                  tf_model.get_variable_names()}
            self.logger.info(f"{k} : {v}")

    def analyze_end_performance(self):
        self.logger.info(f"###### FINAL RESULTS FOR EACH GAME###### \n")
        total_results = [0, 0, 0, 0]
        for game in self.instance_predictions:
            self.logger.info(f"Simulated game {game.home_team.name} vs {game.away_team.name} "
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
                self.logger.info(f"{k} : {v}")
        self.logger.info("\n###### FINAL RESULTS FOR ALL GAMES###### \n")
        final_stats = analyze_confusion_matrix(total_results, sum(total_results))
        for k, v in final_stats.items():
            self.logger.info(f"{k} : {v}")
        self.write_stats_to_json(final_stats, self.outfile)

    def write_stats_to_json(self, stats: Dict, path: str):
        if os.path.isfile(path):
            with open(path, 'r') as json_file:
                prev_data = json.load(json_file)
                json_file.close()
            with open(path, 'w') as json_file:
                stats['season'] = self.season
                stats['num_epochs'] = self.num_epochs
                stats['nn_shape'] = self.nn_shape
                stats['best_performer'] = dict()
                stats['best_performer']['Accuracy'] = self.best_accuracy
                stats['best_performer']['Labels'] = self.labels_used
                stats['best_performer']["Vars"] = self.best_vars
                prev_data[datetime.datetime.now().__str__()] = stats
                json.dump(prev_data, json_file)
        else:
            with open(path, 'w') as json_file:
                stats['season'] = self.season
                stats['num_epochs'] = self.num_epochs
                stats['nn_shape'] = self.nn_shape
                stats['best_performer'] = dict()
                stats['best_performer']['Accuracy'] = self.best_accuracy
                stats['best_performer']['Labels'] = self.labels_used
                stats['best_performer']["Vars"] = self.best_vars
                json.dump({datetime.datetime.now().__str__(): stats}, json_file)


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
