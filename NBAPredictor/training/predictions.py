import datetime
import json
import logging
import os
from typing import Dict, Tuple, List

import tensorflow as tf

from game import Game
from game_period import GamePeriod
from team import Team


class Predictions:
    """
    The Predictions is class is tasked with taking the predicted outcomes from either Tensorflow or sklearn and
    running analysis on the outcomes across all of the epochs. Log important data such as the accuracy and precision
    of each instance in the epoch as well as the average across all epochs at the end.

    At the end, also write to JSON the features used for this program execution, the topology used, the average
    accuracy, precision, etc, along with the best preforming instance for the epoch. This is done so we can track the
    progress of NBAPredictor and record how certain models affect the accuracy of the program. This is very useful in
    the start of this project as the correct features were not known yet so each individual features preformance must
    be recorded

    Attributes
    ----------
    season: str
        The season that was used for training and testing
    outfile: str
        A path for where to write the data about this execution instance of NBAPredictor
    instance_predictions: dict
        A dictionary of Game objects to a List of Teams. The list of teams correspond to which team the model
        predicted to win in that game for each epoch. Note this list should be equal to size num_epochs.
    labels_used: list
        The feature labels used for training and testing
    best_accuracy: float
        The highest accuracy value every recorded in any program execution of NBAPredictor
    best_vars: dict
        A dictionary containing model information about the program execution of NBAPredictor that resulted in the
        highest accuracy.
    num_epochs: int
        The number of epochs or instances the model was trained/tested.
    nn_shape: list
        A list of integers corresponding to the NN topology. For example [22, 10] is a 2 layer Network with 22
        and 10 nodes in the layers.
    logger: logging
        The logger for this class
    svm_compat: bool
        Pass this flag if the SVM model was run instead of the DNN.

    Methods
    -------
    add_dnn_prediction_instance (game: Game, prediction_prob: list)
        Takes as input a prediction instance from Tensorflow for a particular game. Determines if Tensorflow made the
        correct choice or not. Log the prediction it made regardless or right/wrong for this game in the the
        instance_predictions dict.
    add_dnn_seasonal_prediction_instance
        Takes as input a list of prediction from TensorFlow and records the prediction by calling
        add_dnn_prediction_instance. Also keeps track of a confusion matrix for this current instance of runs.
    add_svm_seasons_prediction_instance
        Like add_dnn_seasonal_prediction_instance, but for compatibility woth the SVM model.
    analyze_end_performance
        For each epoch run, build a confusion matrix and display the results of True Positives, True Negatives,
        False Positives, False Negatives, Accuracy, Precision, etc. Also display the average at the end for all
        instances run.
    write_stats_to_json
        Takes the output from analyze_end_performance and writes it to JSON. This way, each instance of NBAPredictor
        is recorded and we can see which ones preformed better than others.
    """

    def __init__(self, season: str, num_epochs: int, nn_shape: List[int], labels_used: List[str], outfile: str,
            svm_compat=False):
        """
        Parameters
        ----------
        season: str
            The season that was used for training and testing
        outfile: str
            A path for where to write the data about this execution instance of NBAPredictor
        instance_predictions: dict
            A dictionary of Game objects to a List of Teams. The list of teams correspond to which team the model
            predicted to win in that game for each epoch. Note this list should be equal to size num_epochs.
        labels_used: list
            The feature labels used for training and testing
        best_accuracy: float
            The highest accuracy value every recorded in any program execution of NBAPredictor
        best_vars: dict
            A dictionary containing model information about the program execution of NBAPredictor that resulted in the
            highest accuracy.
        num_epochs: int
            The number of epochs or instances the model was trained/tested.
        nn_shape: list
            A list of integers corresponding to the NN topology. For example [22, 10] is a 2 layer Network with 22
            and 10 nodes in the layers.
        logger: logging
            The logger for this class
        svm_compat: bool
            Pass this flag if the SVM model was run instead of the DNN.
        """
        self.season = season
        self.outfile = outfile
        self.instance_predictions: Dict[Game, List[Team]] = dict()
        self.labels_used = labels_used
        self.best_accuracy = float('-inf')
        self.best_vars = None
        self.num_epochs = num_epochs
        self.nn_shape = nn_shape
        self.logger = logging.getLogger(f"NBAPredictor.{self.__class__.__name__}")
        self.svm_compat = svm_compat

    def add_dnn_prediction_instance(self, game: Game, prediction_prob: List[float]) -> Tuple[bool, bool]:
        """
        Takes as input a prediction instance from Tensorflow for a particular game. Determines if Tensorflow made the
        correct choice or not. Log the prediction it made regardless or right/wrong for this game in the the
        instance_predictions dict.

        Parameters
        ----------
        game: Game
            The game that was predicted
        prediction_prob: list
            A list of floats, with length 2, wherein the first element is the odds the Home team wins, and the second
            element is the odds the away team wins.

        Returns
        -------
        tuple
            This method returns a tuple of bools, wherein the first element represents the actual winner and the
            second element represents the predicted winner
        """
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

    def add_dnn_seasonal_prediction_instance(self, predictions: List, sorted_games: List[Game],
            tf_model: tf.estimator.DNNClassifier) -> None:
        """
        Takes as input a list of prediction from TensorFlow and records the prediction by calling
        add_dnn_prediction_instance. Also keeps track of a confusion matrix for this current instance of runs.

        Parameters
        ----------
        predictions: list
            List of predictions made for every single game in the testing set
        sorted_games: list
            A sorted list of games that we tested against, in chronological order
        tf_model: tf.estimator.DNNClassifier
            A reference to the model used, so that weight data and biases can be examined.

        Returns
        -------
        None
        """
        confusion_matrix = [0, 0, 0, 0]
        for i, game in enumerate(sorted_games):
            probabilities = predictions[i]['probabilities']
            outcomes = self.add_dnn_prediction_instance(game, probabilities)
            confusion_matrix = recursive_build_confusion_matrix(confusion_matrix, outcomes[0], outcomes[1])
        for k, v in analyze_confusion_matrix(confusion_matrix, len(sorted_games)).items():
            if k == "Accuracy" and v > self.best_accuracy:
                self.best_accuracy = v
                self.best_vars = {name: tf_model.get_variable_value(name).tolist() for name in
                                  tf_model.get_variable_names()}
            self.logger.info(f"{k} : {v}")

    def add_svm_seasons_prediction_instance(self, predictions: List[str], sorted_games: List[Game]) -> None:
        """
        Like add_dnn_seasonal_prediction_instance, but for compatibility woth the SVM model.

        Parameters
        ----------
        predictions: list
            List of predictions made for every single game in the testing set
        sorted_games: list
            A sorted list of games that we tested against, in chronological order

        Returns
        -------
        None
        """
        confusion_matrix = [0, 0, 0, 0]
        for i, game in enumerate(sorted_games):
            self.instance_predictions.setdefault(game, list())
            predicted_outcome = predictions[i]
            predicted_winner = game.home_team if predicted_outcome == 'H' else game.away_team
            predicted_loser = game.away_team if predicted_winner == game.home_team.name else game.home_team
            actual_winner = game.home_team if game.home_team.scores.get(GamePeriod.TOTAL) > game.away_team.scores.get(
                GamePeriod.TOTAL) else game.away_team
            actual_loser = game.away_team if actual_winner is game.home_team else game.home_team
            self.logger.info(f"PREDICTED: {predicted_winner.name} beats {predicted_loser.name} for game {game.code}")
            self.logger.info(f"ACTUAL RESULT: {actual_winner.name} beats {actual_loser.name} for game {game.code}\n")
            self.instance_predictions[game].append(predicted_winner)
            confusion_matrix = recursive_build_confusion_matrix(confusion_matrix,
                                                                True if actual_winner == game.home_team else False,
                                                                True if predicted_winner == game.home_team else False)
        for k, v in analyze_confusion_matrix(confusion_matrix, len(sorted_games)).items():
            if k == "Accuracy" and v > self.best_accuracy:
                self.best_accuracy = v
            self.logger.info(f"{k} : {v}")

    def analyze_end_performance(self) -> None:
        """
        For each epoch run, build a confusion matrix and display the results of True Positives, True Negatives,
        False Positives, False Negatives, Accuracy, Precision, etc. Also display the average at the end for all
        instances run

        Returns
        -------
        None
        """
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

    def write_stats_to_json(self, stats: Dict, path: str) -> None:
        """
        Takes the output from analyze_end_performance and writes it to JSON. This way, each instance of NBAPredictor
        is recorded and we can see which ones preformed better than others

        Parameters
        ----------
        stats: dict
            Data dump of this entire instance, created by analyze_end_performance
        path: str
            File path of where to write the results to.

        Returns
        -------
        None
        """
        if os.path.isfile(path):
            with open(path, 'r') as json_file:
                prev_data = json.load(json_file)
                json_file.close()
            with open(path, 'w') as json_file:
                stats['season'] = self.season
                stats['num_epochs'] = self.num_epochs
                if not self.svm_compat:
                    stats['model_used'] = "DNN"
                    stats['nn_shape'] = self.nn_shape
                else:
                    stats['model_used'] = "SVM-SVC"
                stats['best_performer'] = dict()
                stats['best_performer']['Accuracy'] = self.best_accuracy
                stats['best_performer']['Labels'] = self.labels_used
                if not self.svm_compat:
                    stats['best_performer']["Vars"] = self.best_vars
                prev_data[datetime.datetime.now().__str__()] = stats
                json.dump(prev_data, json_file)
        else:
            with open(path, 'w') as json_file:
                stats['season'] = self.season
                stats['num_epochs'] = self.num_epochs
                if not self.svm_compat:
                    stats['model_used'] = "DNN"
                    stats['nn_shape'] = self.nn_shape
                else:
                    stats['model_used'] = "SVM-SVC"
                stats['best_performer'] = dict()
                stats['best_performer']['Accuracy'] = self.best_accuracy
                stats['best_performer']['Labels'] = self.labels_used
                if not self.svm_compat:
                    stats['best_performer']["Vars"] = self.best_vars
                json.dump({datetime.datetime.now().__str__(): stats}, json_file)


def recursive_build_confusion_matrix(previous: List[int], actual_home_win: bool, predicted_home_win: bool) -> List[int]:
    """
    A utility function that allows a Confusion Matrix to be built over time. The function expects as input a previous
    state of the confusion matrix, which is represented as list with the following format:
     [ True Positive, False Negative, False Positive, True Negative]
    and a new instance to add, meaning an actual result, and a predicted result.

    Parameters
    ----------
    previous: list
        Previous state of the confusion matrix
    actual_home_win: bool
        Whether the home team actually won
    predicted_home_win: bool
        Whether the home team was predicted to win

    Returns
    -------
    list
        The state of the confusion matrix
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
    """
    Utility function to analyze a confusion matrix.  This function will count the number of true positives,
    true negatives, false positives, and false negatives. It will return a dict with those numbers along with the
    accuracy, precision, and error rate.

    Parameters
    ----------
    confusion_matrix: list
        List representation of a confusion matrix
    num_games: int
        The number of games there were in the testing set.

    Returns
    -------
    dict
        Stats including: number of true positives,
    true negatives, false positives, and false negatives and accuracy, precision, and error rate
    """
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
