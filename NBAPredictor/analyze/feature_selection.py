import random
from typing import List

from read_game import DEFAULT_FEATURES, POSSIBLE_FEATURES
from read_stats import ReadStats


class FeatureSelection:

    def __init__(self, read_stats: ReadStats, strategy: str):
        self.read_stats = read_stats
        if strategy == "UsePrevious":
            self.features = read_stats.get_last_used_features()
        elif strategy == "UseNBestAndRandom":
            self.features = self.use_n_best_strategy()
        elif strategy == "UseBest":
            self.features = self.use_best()
        else:
            self.features = DEFAULT_FEATURES

    def use_n_best_strategy(self, n: int = 6, feature_set_size=12, vary_input_size=False) -> List[str]:
        """
        This strategy picks the (n/2) best features for the positive weights and the (n/2) best features for the
        negatives weights and uses them in the current feature set. The other features are selected randomly from a
        list of all features
        :return:
        """
        if n % 2 != 0:
            print(f"WARNING: Selected N: {n} is not even so an even split cant be made")
            n += 1
        assert n + 1 < feature_set_size
        features = [x[0] for x in self.read_stats.best_features[:int(n / 2)]] + [y[0] for y in
                                                                                 self.read_stats.best_features[::-1][
                                                                                 :int(n / 2)]]
        while len(features) <= feature_set_size:
            choice = POSSIBLE_FEATURES[random.randint(0, len(POSSIBLE_FEATURES))]
            if choice not in features:
                features.append(choice)
        return features

    def use_best(self, feature_set_size=12, vary_input_size=False) -> List[str]:
        if feature_set_size % 2 != 0:
            feature_set_size += 1
        return [x[0] for x in self.read_stats.best_features[:int(feature_set_size / 2)]] + [y[0] for y in
                                                                                            self.read_stats.best_features[
                                                                                            ::-1][
                                                                                            :int(feature_set_size / 2)]]
