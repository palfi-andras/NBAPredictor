from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time
from typing import List, Union

import tensorflow as tf
from sklearn import svm

tf.logging.set_verbosity(tf.logging.ERROR)

from league import League
from predictions import Predictions
from read_game import ReadGames
from predict_next_season import PredictNextSeason, Playoffs

DEFAULT_METHOD = "DNN"


class TensorflowOperations:
    """
    The TensorflowOperations class is a class that NBAPredictor uses as a facade into various Machine Learning
    libraries such as sklearn and TensorFlow. This module runs our entire training and testing routines before
    passing off control to the Predictions class to organize results data.

    Attributes
    ----------
    mode: str
        What learning mode we should run the NBAPredictor in for this execution. Options are "DNN" or "SVM"
    league: League
        The main League object
    num_epochs: int
        How many epochs to run this configuration for.
    learning_rate: float
        The weight at which to update weights in TensorFlow
    nn_shape: list
        A list of integers corresponding to the NN topology. For example [22, 10] is a 2 layer Network with 22
        and 10 nodes in the layers.
    model_dir: str
        A path to a master directory of models. Individual model directories for various configurations will all
        be placed into this directory
    normalize_weights: bool
        If set to true, a weight column will be added onto the training and testing examples to boost or downplay
        certain instances
    parsed_season: ReadGames
        A ReadGames instance. This object contains all of the training/testing features and labels required to run
        any of the models in TensorflowOperations.
    feature_cols: list
        A list of feature columns for this instance
    model: Tensorflow DNN Classifier or Scikit SVM model.
        An ML model configured according to the parameters requested at object initialization.
    predictions: Predictions
        A predictions object to use after all training/testing has concluded. The Predictions class is in charge of
        post-analysis.
    train_input_function: tf.estimator.inputs.numpy_input_fn
        An input function to use for training
    test_input_function: tf.estimator.inputs.numpy_input_fn
        An input function to use for testing
    logger: logging
        A logger for this class

    Methods
    -------
    create_feature_columns
        Creates a list of feature columns from the features passed in on initialization
    run
        Higher-level function that runs the NBAPredictor for this instance, for the requested number of epochs.
        "Runs" in this instance refers to training, testing, and evaluating.
    train
        In the case of DNN, the model will be trained against the training features in the parsed_season object. For
        SVM strategy, the model will be fit against the training features.
    evaluate
        Used by the TensorFlow DNN in order to evaluate the DNN against the test input function.
    get_predictions
        This is the method tasked with testing against the trained samples. For each game in the testing data,
        make a Home or Away Win/Loss prediction. Predictions are expected to be in the format of a tuple of floats
        such as [0.45, 0.55], where the first index is the odds the home team wins and the second index is the odds
        the away team wins. This list of predictions is passed onto the Predictions object for further post-analysis.
    """

    def __init__(self, league: League, num_epochs: int, learning_rate: float, nn_shape: List[int], season: str,
            split: float, outfile: str, model_dir: str, features: List[str], cache_dir: str, mode: str = DEFAULT_METHOD,
            normalize_weights: bool = False, cache_numpy_structures=False, predict_next_season=False,
            next_season_csv: str = None):
        """
        Parameters
        ----------
        league: League
            The main League object
        num_epochs: int
            How many epochs to run this configuration for.
        learning_rate: float
            The weight at which to update weights in TensorFlow
        nn_shape: list
            A list of integers corresponding to the NN topology. For example [22, 10] is a 2 layer Network with 22
            and 10 nodes in the layers.
        season: str
            The season or seasons that we are preforming training/testing on
        split: float
            The split at which to divide training/testing data
        outfile: str
            A path to a file, where data about each program execution will be dumped, effectivley serving as a
            history of previous models.
        model_dir: str
            A path to a master directory of models. Individual model directories for various configurations will all
            be placed into this directory
        features: list
            A list of features to use as inputs for the NN or SVM.
        logger: logging
            A logger for this class
        mode: str
            What learning mode we should run the NBAPredictor in for this execution. Options are "DNN" or "SVM"
        normalize_weights: bool
            If set to true, a weight column will be added onto the training and testing examples to boost or downplay
            certain instances
        cache_numpy_structures: bool
            If set to True, the NumPy array from this object will be cached as a pickle object so that if the same
            dataset is used again we do not need to prefrom all the parsing and normalizations.
        predict_next_season: bool
            If set to True, this will not test against existing Data, but instead try to predict the upcoming
            2019-2020 SEASON NBA
         next_season_csv: str
            Path to next seasons CSV file
        """
        self.logger = logging.getLogger(f"NBAPredictor.{self.__class__.__name__}")
        self.mode = mode
        self.leauge = league
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.nn_shape = nn_shape
        self.model_dir = model_dir
        self.normalize_weights = normalize_weights
        self.predict_next_season = predict_next_season
        if self.predict_next_season and self.num_epochs > 1:
            self.logger.info(
                f"Predict Next Season mode can run only for one epoch. Overriding value of {self.num_epochs}")
            self.num_epochs = 1
        if self.mode == "SVM":
            self.parsed_season = ReadGames(self.leauge, season, split, cache_dir, features, svm_compat=True,
                                           normalize_weights=self.normalize_weights, cache=cache_numpy_structures)
        else:
            self.parsed_season = ReadGames(self.leauge, season, split, cache_dir, features, svm_compat=False,
                                           normalize_weights=self.normalize_weights,
                                           cache=cache_numpy_structures) if not self.predict_next_season else \
                PredictNextSeason(
                next_season_csv=next_season_csv, leauge=league, season=season, split=split, cache_dir=cache_dir,
                features=features, cache=cache_numpy_structures)
        self.feature_cols = self.create_feature_columns()
        self.model = self.create_model()
        if self.mode == "SVM":
            self.predictions = Predictions(season, num_epochs, nn_shape, self.parsed_season.features, outfile,
                                           svm_compat=True)
        else:
            self.predictions = Predictions(season, num_epochs, nn_shape, self.parsed_season.features, outfile,
                                           svm_compat=False)
        self.train_input_function = tf.estimator.inputs.numpy_input_fn(x=self.parsed_season.training_features,
                                                                       y=self.parsed_season.training_labels,
                                                                       batch_size=500, num_epochs=None, shuffle=False)
        self.test_input_function = tf.estimator.inputs.numpy_input_fn(x=self.parsed_season.testing_features,
                                                                      y=self.parsed_season.testing_labels, num_epochs=1,
                                                                      shuffle=False)
        if self.mode == "SVM":
            self.logger.info(f"Running SVC on the {season} NBA Season(s) for "
                             f"{self.num_epochs} epochs with the following input features used: "
                             f"{self.parsed_season.features}")
        else:
            self.logger.info(f"Running DNN on the {season} NBA Season(s) for"
                             f" {self.num_epochs} "
                             f"epochs with the following NN shape: {self.nn_shape} and the following input "
                             f"features: "
                             f"{self.parsed_season.features}")

    def create_feature_columns(self) -> List[tf.feature_column.numeric_column]:
        """
        Creates a list of feature columns from the features passed in on initialization

        Returns
        -------
        list
            A list of feature columns for each item in the ReadGame object feature attribute
        """
        return [tf.feature_column.numeric_column(key=item) for item in self.parsed_season.features]

    def create_model(self) -> Union[tf.estimator.DNNClassifier, svm.SVC]:
        if self.mode == "DNN":
            if self.normalize_weights:
                return tf.estimator.DNNClassifier(model_dir=self.model_dir, hidden_units=self.nn_shape,
                                                  feature_columns=self.feature_cols, n_classes=2,
                                                  label_vocabulary=['H', 'A'],
                                                  weight_column=tf.feature_column.numeric_column('weight'),
                                                  optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
                                                      learning_rate=self.learning_rate,
                                                      l1_regularization_strength=0.001))
            else:
                return tf.estimator.DNNClassifier(model_dir=self.model_dir, hidden_units=self.nn_shape,
                                                  feature_columns=self.feature_cols, n_classes=2,
                                                  label_vocabulary=['H', 'A'],
                                                  optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
                                                      learning_rate=self.learning_rate,
                                                      l1_regularization_strength=0.01))
        elif self.mode == "SVM":
            return svm.SVC(kernel='rbf')
        else:
            raise RuntimeError(f"{self.mode} is not an implemented or recognized ML strategy in NBAPredictor")

    def run(self) -> None:
        """
        Higher-level function that runs the NBAPredictor for this instance, for the requested number of epochs.
        "Runs" in this instance refers to training, testing, and evaluating.

        Returns
        -------
        None
        """
        if self.mode == "DNN":
            if not self.predict_next_season:
                for x in range(0, self.num_epochs):
                    self.logger.info(f"Running instance #{x + 1}")
                    self.train()
                    self.evaluate()
                    self.get_predictions()
            else:
                self.logger.info(f"Predicting NBA Season for 2019-2020")
                self.train()
                self.evaluate()
                playoffs: Playoffs = self.get_predictions()
                while playoffs.generate_test_data_for_playoffs():
                    testing_features, testing_labels = playoffs.generate_test_data_for_playoffs()
                    self.test_input_function = tf.estimator.inputs.numpy_input_fn(x=testing_features, y=testing_labels,
                                                                                  num_epochs=1, shuffle=False)
                    self.evaluate()
                    predictions = list(self.model.predict(input_fn=self.test_input_function))
                    playoffs.record_playoff_results(predictions)
                playoffs.log_playoff_results()
                time.sleep(8)
        elif self.mode == "SVM":
            for x in range(0, self.num_epochs):
                self.logger.info(f"Running instance #{x + 1}")
                self.train()
                self.get_predictions()
        else:
            raise RuntimeError(f"{self.mode} is not an implemented or recognized ML strategy in NBAPredictor")
        self.predictions.analyze_end_performance() if not self.predict_next_season else None

    def train(self) -> None:
        """
        In the case of DNN, the model will be trained against the training features in the parsed_season object. For
        SVM strategy, the model will be fit against the training features.

        Returns
        -------
        None
        """
        if self.mode == "DNN":
            self.model.train(input_fn=self.train_input_function, steps=100)
        elif self.mode == "SVM":
            self.model.fit(self.parsed_season.training_features, self.parsed_season.training_labels)
        else:
            raise RuntimeError(f"{self.mode} is not an implemented or recognized ML strategy in NBAPredictor")

    def evaluate(self) -> None:
        """
        Used by the TensorFlow DNN in order to evaluate the DNN against the test input function

        Returns
        -------
        None
        """
        self.model.evaluate(input_fn=self.test_input_function)

    def get_predictions(self) -> Union[None, Playoffs]:
        """
        This is the method tasked with testing against the trained samples. For each game in the testing data,
        make a Home or Away Win/Loss prediction. Predictions are expected to be in the format of a tuple of floats
        such as [0.45, 0.55], where the first index is the odds the home team wins and the second index is the odds
        the away team wins. This list of predictions is passed onto the Predictions object for further post-analysis

        Returns
        -------
        None
        """
        if self.mode == "DNN":
            if not self.predict_next_season:
                predictions = list(self.model.predict(input_fn=self.test_input_function))
                start_index = self.parsed_season.training_size + 1
                predicted_games = [self.parsed_season.sorted_games[i] for i in
                                   range(start_index, len(self.parsed_season.sorted_games))]
                self.predictions.add_dnn_seasonal_prediction_instance(predictions, predicted_games, self.model)
            else:
                predictions = list(self.model.predict(input_fn=self.test_input_function))
                return self.parsed_season.analyze_end_of_season_predictions(predictions)
        elif self.mode == "SVM":
            predictions = self.model.predict(self.parsed_season.testing_features)
            start_index = self.parsed_season.training_size + 1
            predicted_games = [self.parsed_season.sorted_games[i] for i in
                               range(start_index, len(self.parsed_season.sorted_games))]
            self.predictions.add_svm_seasons_prediction_instance(predictions, predicted_games)
        else:
            raise RuntimeError(f"{self.mode} is not an implemented or recognized ML strategy in NBAPredictor")
