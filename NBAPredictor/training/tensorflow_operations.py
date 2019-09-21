from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List

import tensorflow as tf

from league import League
from read_game import ReadGames, build_input_labels_array

TRAIN_SIZE = 0.9


class TensorflowOperations:

    def __init__(self, league: League, num_epochs=100, layers: List[int] = [30], learning_rate=0.01):
        self.leauge = league
        self.num_epochs = num_epochs
        self.layers = layers
        self.learning_rate = learning_rate
        self.feature_cols = self.create_feature_columns()
        self.model = self.create_model()
        self.parsed_season = ReadGames(self.leauge)
        self.train_input_function = tf.estimator.inputs.numpy_input_fn(x=self.parsed_season.training_features,
                                                                       y=self.parsed_season.training_labels,
                                                                       batch_size=500, num_epochs=None, shuffle=False)
        self.test_input_function = tf.estimator.inputs.numpy_input_fn(x=self.parsed_season.testing_features,
                                                                      y=self.parsed_season.testing_labels, num_epochs=1,
                                                                      shuffle=False)

    def create_feature_columns(self):
        feature_cols = list()
        for item in build_input_labels_array():
            feature_cols.append(tf.feature_column.numeric_column(key=item))
        return feature_cols

    def create_model(self):
        return tf.estimator.DNNClassifier(model_dir='model/', hidden_units=[5], feature_columns=self.feature_cols,
                                          n_classes=2, label_vocabulary=['H', 'A'],
                                          optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
                                              learning_rate=self.learning_rate, l1_regularization_strength=0.001))

    def train(self):
        self.model.train(input_fn=self.train_input_function, steps=self.num_epochs)

    def evaluate(self):
        self.model.evaluate(input_fn=self.test_input_function)

    def get_predictions(self):
        predictions = list(self.model.predict(input_fn=self.test_input_function))
        for i in range(0, len(predictions)):
            game = self.parsed_season.sorted_games[i]
            home_team = game.home_team.name
            away_team = game.away_team.name
            actual_result = self.parsed_season.testing_labels[i]
            probabilities = predictions[i]['probabilities']
            if probabilities[0] > probabilities[1]:
                print(f"PREDICTION: {home_team} beats {away_team}")
            else:
                print(f"PREDICTION: {away_team} beats {home_team}")
            if actual_result == 'H':
                print(f"ACTUAL RESULT: {home_team} beat {away_team}")
            elif actual_result == 'A':
                print(f"ACTUAL RESULT: {away_team} beat {home_team}")
            else:
                raise Exception(f"Unrecognized value {actual_result}, must be either H for Home Win or A for Away Win")
