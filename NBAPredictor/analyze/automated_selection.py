import hashlib
import random
from typing import List

from read_game import DEFAULT_FEATURES, POSSIBLE_FEATURES
from read_stats import ReadStats


class AutomatedSelection:
    """
    NBAPredictor has the ability to run in a fairly automated fashion, executing "batch" runs of itself,
    each with different parameters. For these runs, it can swap out features with other features in order to expose
    the ML models to as many different feature sets as possible.

    This class is called before any Machine Learning activity occurs in NBAPredictor, and it determines which
    features from the pool of all features will be used for this current execution instance, and what the Neural Network
    topology will look like for this instance. It also uses hashlib to create a unique string representation of the
    features and topology used for this instance. This is done so that this string may be passed to TensorFlow,
    and if we run NBAPredictor with the same exact parameters again, the same model string will be generated and
    passed to TensorFlow and it will re-use its model from those previous instances.

    Attributes
    ----------
    read_stats: ReadStats
        A ReadStats object that is used for gathering information on the best performing features
    strategy: str
        The strategy to use for Feature Selection
    nn_shape: list
        A list of integers representing the Neural Network shape. Therefore a list of [20, 3] would be a NN with 2
        hidden layers, 20 and 3 nodes respectivley.
    features: list
        A list of features that represent what the feature subset will be for this current execution instance of
        NBAPredictor
    model_name: str
        A unique string generated from the names of the features used and the neural network topolgy that is used to
        give this feature a unique identifier in order to recreate model data for TensorFlow

    Methods
    -------
    create_model_name
        Creates a unique string representation of this model. The string is generated using a sha1 hash of the
        feature names and NN shape/topology so that a future execution of this program with similar features is
        forced to re-used past training data.
    use_n_best_strategy (n: int = 6, feature_set_size=12, vary_input_size=True)
        This feature selection strategy goes through the set of features used previously in NBAPredictor and picks
        the N best performers. It then populates the rest of the feature set making random choices from the pool of
        all features.
    use_best (self, feature_set_size=10, vary_input_size=False)
        This feature selection strategy goes through the set of features used previously in NBAPredictor and
        populates the list of features to be used for this current execution instance with the best performers
        historically.
    generate_random_nn_shape
        Generates a random Neural Network shape bounded between 2-4 layers 0-20 neurons at each layers.

    """

    def __init__(self, read_stats: ReadStats, strategy: str, nn_shape: List[int] = None):
        """
        Parameters
        ----------
        read_stats: ReadStats
            A ReadStats object that is used for gathering information on the best performing features
        strategy: str
            The strategy to use for Feature Selection
        nn_shape: list
            A list of integers representing the Neural Network shape. Therefore a list of [20, 3] would be a NN with 2
            hidden layers, 20 and 3 nodes respectivley.
        """
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

    def create_model_name(self) -> str:
        """
        Creates a unique string representation of this model. The string is generated using a sha1 hash of the
        feature names and NN shape/topology so that a future execution of this program with similar features is
        forced to re-used past training data. Note, if the same features and NN Shape is used, this method will
        recreate the same exact string.

        Returns
        -------
        name: str
            The unique model name of this instance
        """
        name = "model-"
        assert self.nn_shape
        for layer in self.nn_shape:
            name += f"{layer}-"
        assert self.features
        name += str(hashlib.sha1(str(self.features).encode('utf-8')).hexdigest())
        return name

    def use_n_best_strategy(self, n: int = 6, feature_set_size=12, vary_input_size=True) -> List[str]:
        """
        This feature selection strategy goes through the set of features used previously in NBAPredictor and picks
        the N best performers. It then populates the rest of the feature set making random choices from the pool of
        all features.

        Parameters
        ----------
        n: int
            The amount of best features to use in the next feature subset
        feature_set_size: int
            The total size of the feature subset
        vary_input_size: bool
            If set to true, this will vary the size of the actual feature set, +/- 4. This is done in case different
            sizes of feature subsets want to be tested out.

        Returns
        -------
        features: list
            A list of feature names to use for this current execution instance of NBAPredictor
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
        """
        This feature selection strategy goes through the set of features used previously in NBAPredictor and
        populates the list of features to be used for this current execution instance with the best performers
        historically.

        Parameters
        ----------
        feature_set_size: int
            The total size of the feature subset
        vary_input_size: bool
            If set to true, this will vary the size of the actual feature set, +/- 4. This is done in case different
            sizes of feature subsets want to be tested out.

        Returns
        -------
        list
            A list of feature names to use for this current execution instance of NBAPredictor
        """
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
        """
        Generates a random Neural Network shape .

        Returns
        -------
        layers: list
            List of a Neural Network shape using the syntax defined above in the top level class doc.
        """
        return [random.randint(8, 50) for x in range(random.randint(2, 5))]
