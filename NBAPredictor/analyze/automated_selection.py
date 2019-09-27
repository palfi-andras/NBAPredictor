import hashlib
import random
from typing import List

from read_game import DEFAULT_FEATURES, POSSIBLE_FEATURES
from read_stats import ReadStats

"""
A class that can make automatic adjustments to the NN shape and Features used to help automate the testing of different
configuartions
"""


class AutomatedSelection:

    def __init__(self, read_stats: ReadStats, strategy: str, nn_shape: List[int] = None):
        self.read_stats = read_stats
        if nn_shape is not None:
            self.nn_shape = nn_shape
        else:
            self.nn_shape = self.generate_random_nn_shape()
        if strategy == "UsePrevious":
            self.features = read_stats.get_last_used_features()
        elif strategy == "UseNBestAndRandom":
            self.features = self.use_n_best_strategy()
        elif strategy == "UseBest":
            self.features = self.use_best()
        else:
            self.features = DEFAULT_FEATURES
        self.model_name = self.create_model_name()

    def create_model_name(self):
        name = "model-"
        for layer in self.nn_shape:
            name += f"{layer}-"
        assert self.features
        name += str(hashlib.sha1(str(self.features).encode('utf-8')).hexdigest())
        return name

    def use_n_best_strategy(self, n: int = 6, feature_set_size=12, vary_input_size=True) -> List[str]:
        """
        This strategy picks the (n/2) best features for the positive weights and the (n/2) best features for the
        negatives weights and uses them in the current feature set. The other features are selected randomly from a
        list of all features
        :return:
        """
        if n % 2 != 0:
            n += 1
        if vary_input_size:
            # This will play around with the feature set size, +/- 4
            feature_set_size = feature_set_size - random.randint(-4, 4)
        # We want to have at least 2 "new" features
        assert n + 1 < feature_set_size
        features = [x[0] for x in self.read_stats.best_features[:int(n / 2)]] + [y[0] for y in
                                                                                 self.read_stats.best_features[::-1][
                                                                                 :int(n / 2)]]
        # Add new features randomly
        while len(features) < feature_set_size:
            choice = POSSIBLE_FEATURES[random.randint(0, len(POSSIBLE_FEATURES) - 1)]
            if choice not in features:
                features.append(choice)
        return features

    def use_best(self, feature_set_size=10, vary_input_size=False) -> List[str]:
        if vary_input_size:
            # This will play around with the feature set size, +/- 4
            feature_set_size = feature_set_size - random.randint(-4, 4)
        if feature_set_size % 2 != 0:
            feature_set_size += 1
        return [x[0] for x in self.read_stats.best_features[:int(feature_set_size / 2)]] + [y[0] for y in
                                                                                            self.read_stats.best_features[
                                                                                            ::-1][
                                                                                            :int(feature_set_size / 2)]]

    def generate_random_nn_shape(self):
        layers = []
        num_layers = random.randint(2, 4)
        for layer in range(num_layers):
            if layer == 0:
                layers.append(random.randint(12, 24))
            elif layer == 1:
                layers.append(random.randint(6, 10))
            else:
                layers.append(random.randint(2, 4))
        return layers
