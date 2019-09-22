from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List

import tensorflow as tf

from league import League
from predictions import Predictions
from read_game import ReadGames, build_input_labels_array

TRAIN_SIZE = 0.9
NUM_EPOCHS = 100
NEURAL_NETWORK_SHAPE = [16, 5, 3]


class TensorflowOperations:

    def __init__(self, league: League, predictions: Predictions, num_epochs=NUM_EPOCHS, learning_rate=0.01,
            nn_shape: List[int] = NEURAL_NETWORK_SHAPE):
        self.leauge = league
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.feature_cols = self.create_feature_columns()
        self.nn_shape = nn_shape
        self.model = self.create_model()
        self.parsed_season = ReadGames(self.leauge)
        self.predictions = predictions
        self.predictions.set_season(self.parsed_season.sorted_games[0].season)
        self.predictions.set_num_epochs(self.num_epochs)
        self.predictions.set_nn_shape(self.nn_shape)
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
        return tf.estimator.DNNClassifier(model_dir='model/', hidden_units=self.nn_shape,
                                          feature_columns=self.feature_cols, n_classes=2, label_vocabulary=['H', 'A'],
                                          optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
                                              learning_rate=self.learning_rate, l1_regularization_strength=0.001))

    def run_neural_network(self):
        for x in range(0, self.num_epochs):
            print(f"Running instance #{x}")
            self.train()
            self.evaluate()
            self.get_predictions()
        self.predictions.analyze_end_performance()
        self.predictions.save_predictions_instance()

    def train(self):
        self.model.train(input_fn=self.train_input_function, steps=self.num_epochs)

    def evaluate(self):
        self.model.evaluate(input_fn=self.test_input_function)

    def get_predictions(self):
        predictions = list(self.model.predict(input_fn=self.test_input_function))
        start_index = self.parsed_season.training_size + 1
        predicted_games = list()
        for i in range(start_index, len(self.parsed_season.sorted_games)):
            predicted_games.append(self.parsed_season.sorted_games[i])
        self.predictions.add_seasonal_prediction_instance(predictions, predicted_games, self.model)
