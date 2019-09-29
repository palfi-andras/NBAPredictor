from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List
import logging

import tensorflow as tf
from sklearn import svm

tf.logging.set_verbosity(tf.logging.ERROR)

from league import League
from predictions import Predictions
from read_game import ReadGames

DEFAULT_METHOD = "DNN"


class TensorflowOperations:

    def __init__(self, league: League, num_epochs: int, learning_rate: float, nn_shape: List[int], season: str,
            split: float, outfile: str, model_dir: str, features: List[str], logger: logging,
            mode: str = DEFAULT_METHOD):
        self.mode = mode
        self.leauge = league
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.nn_shape = nn_shape
        self.model_dir = model_dir
        if self.mode == "SVM":
            self.parsed_season = ReadGames(self.leauge, season, split, logger, features, svm_compat=True)
        else:
            self.parsed_season = ReadGames(self.leauge, season, split, logger, features, svm_compat=False)
        self.feature_cols = self.create_feature_columns()
        self.model = self.create_model()
        if self.mode == "SVM":
            self.predictions = Predictions(season, num_epochs, nn_shape, self.parsed_season.features, outfile,
                                           logger=logger, svm_compat=True)
        else:
            self.predictions = Predictions(season, num_epochs, nn_shape, self.parsed_season.features, outfile,
                                           logger=logger, svm_compat=False)
        self.train_input_function = tf.estimator.inputs.numpy_input_fn(x=self.parsed_season.training_features,
                                                                       y=self.parsed_season.training_labels,
                                                                       batch_size=500, num_epochs=None, shuffle=False)
        self.test_input_function = tf.estimator.inputs.numpy_input_fn(x=self.parsed_season.testing_features,
                                                                      y=self.parsed_season.testing_labels, num_epochs=1,
                                                                      shuffle=False)
        self.logger = logger

    def create_feature_columns(self):
        feature_cols = list()
        for item in self.parsed_season.features:
            feature_cols.append(tf.feature_column.numeric_column(key=item))
        return feature_cols

    def create_model(self):
        if self.mode == "DNN":
            return tf.estimator.DNNClassifier(model_dir=self.model_dir, hidden_units=self.nn_shape,
                                              feature_columns=self.feature_cols, n_classes=2,
                                              label_vocabulary=['H', 'A'],
                                              optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
                                                  learning_rate=self.learning_rate, l1_regularization_strength=0.001))
        elif self.mode == "SVM":
            return svm.SVC(kernel='rbf')
        else:
            raise NotImplementedError(f"{self.mode} is not an implemented or recognized ML strategy in NBAPredictor")

    def run(self):
        if self.mode == "DNN":
            for x in range(0, self.num_epochs):
                self.logger.info(f"Running instance #{x + 1}")
                self.train()
                self.evaluate()
                self.get_predictions()
        elif self.mode == "SVM":
            for x in range(0, self.num_epochs):
                self.logger.info(f"Running instance #{x + 1}")
                self.train()
                self.get_predictions()
        else:
            raise NotImplementedError(f"{self.mode} is not an implemented or recognized ML strategy in NBAPredictor")
        self.predictions.analyze_end_performance()

    def train(self):
        if self.mode == "DNN":
            self.model.train(input_fn=self.train_input_function, steps=self.num_epochs)
        elif self.mode == "SVM":
            self.model.fit(self.parsed_season.training_features, self.parsed_season.training_labels)
        else:
            raise NotImplementedError(f"{self.mode} is not an implemented or recognized ML strategy in NBAPredictor")

    def evaluate(self):
        self.model.evaluate(input_fn=self.test_input_function)

    def get_predictions(self):
        if self.mode == "DNN":
            predictions = list(self.model.predict(input_fn=self.test_input_function))
            start_index = self.parsed_season.training_size + 1
            predicted_games = [self.parsed_season.sorted_games[i] for i in
                               range(start_index, len(self.parsed_season.sorted_games))]
            self.predictions.add_dnn_seasonal_prediction_instance(predictions, predicted_games, self.model)
        elif self.mode == "SVM":
            predictions = self.model.predict(self.parsed_season.testing_features)
            start_index = self.parsed_season.training_size + 1
            predicted_games = [self.parsed_season.sorted_games[i] for i in
                               range(start_index, len(self.parsed_season.sorted_games))]
            self.predictions.add_svm_seasons_prediction_instance(predictions, predicted_games)
        else:
            raise NotImplementedError(f"{self.mode} is not an implemented or recognized ML strategy in NBAPredictor")
