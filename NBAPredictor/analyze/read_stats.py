import json
import os
import re
from collections import OrderedDict
from typing import Tuple, Dict, Any, Set, List
import logging


class ReadStats:
    """
    This class includes functionality to read the predictions.json file that is used to store data about the previous
    runs or usages of NBAPredictor. This is used to examine which of the past runs have been the most effective and what
    sorts of parameters (Neural Network Topology, features used, weights, etc) were used so that NBAPredictor is
    automatically able to calculate the most optimal type of approach given the history of previous executions. This
    class is usually run before any Machine Learning begins in order to make sure the features we are using really
    are the most ideal ones. If this class is able to find better feature subsets to use, it will suggest so to the
    handler.

    This class also is the sole action executed when NBApredictor is run in 'Analyze' mode. Analyze mode does not
    execute any Training or Testing, it merely analyzes the history of all previous runs (every single configuration
    and training/testing data from NBAPredictor has been recorded) and displays some information on best features,
    accuracies, precision scores, etc.

    This class also maintains a separate JSON file named 'features.json' wherein it records the average initial
    weight used for any of the features selected as input for learning. THis file is statically grown over time
    throughout the lifetime of this program and for it to be reset, the file should be deleted. The purpose of this
    file is to show the average weight value of a particular feature to determine how useful it has been in
    calculating the correct outcome over all the various configuartions it has been used in.


    Attributes
    ----------
    stats_file: str
        A path to the JSON file that contains configuration histories about the previous executions of this program
    feature_file: str
        A path to the features.json file that stores average feature weight values
    logger: logging
        A logger object for this class
    features: dict
        A dictionary of features used historically by NBAPredictor and their average initial weight over time
    best_features: list
        A list of sorted tuples of format ( <Feature Name> , Average Weight Value) sorted from highest average
        feature weight to lowest.


    Methods
    -------
    get_last_used_features
        Find the set of features used in the last execution of NBAPredictor and returns it
    get_highest_accuracy
        Loop through each previous execution of NBAPredictor and find the instance that resulted in the best accuracy
    log_best_performer
        Log data such as features used, accuracies, precision, weights and biases, of the best preforming instance in
        NBAPredictor
    get_highest_precision
        Loop through each previous execution of NBAPredictor and find the instance that resulted in the best precision
    extract_biases (data: dict)
        Creates and returns a  dictionary of the biases used at each neuron in the neural network for a previous
        execution instance of NBAPredictor
    extract_weights (data: dict)
        Creates and returns a dictionary of every feature used in some previous execution of NBAPredictor and what
        the initial weights were at each neuron along with the average at each neuron.
    write_avg_feature_weight
        Updates the 'features.json' file by calculating the average initial weight value for each feature ever used
        in previous NBAPredictor executions.
    get_best_features
        Creates a list of sorted tuples of format ( <Feature Name> , Average Weight Value) sorted from highest average
        feature weight to lowest and returns it.


    """

    def __init__(self, stats_file: str, feature_file: str, logger: logging):
        """
        Parameters
        ----------
        stats_file: str
            A path to the JSON file that contains configuration histories about the previous executions of this program
        feature_file: str
            A path to the features.json file that stores average feature weight values
        logger: logging
            A logger object for this class
        """
        assert os.path.isfile(stats_file), f"Cant find file {stats_file}"
        with open(stats_file, 'r') as json_file:
            self.stats = json.load(json_file)
        self.feature_file = feature_file
        self.features = self.write_avg_feature_weight()
        self.best_features = self.get_best_features()
        self.logger = logger
        self.logger.info(
            f"This program has been run {len(self.stats)} times.\n These are the 3 best features so far in predicting "
            f"a home team win correctly: ")
        for index, feature in enumerate(self.best_features[:3]):
            self.logger.info(f"{index + 1}. {feature[0]}, Average Weight: {feature[1]}")
        self.logger.info("\nThese are the 3 best features so far in predicting an away team victory correctly")
        for index, feature in enumerate(self.best_features[::-1][:3]):
            self.logger.info(f"{index + 1}. {feature[0]}, Average Weight: {feature[1]}")

    def get_last_used_features(self) -> Set[str]:
        """
        Find the set of features used in the last execution of NBAPredictor and return it

        Returns
        -------
        set
           A set of features used in the last instance of NBAPredictor
        """
        return self.stats[next(reversed(OrderedDict(self.stats)))]['best_performer']['Labels']

    def get_highest_accuracy(self) -> Tuple[float, Dict[str, Any]]:
        """
        Loop through each previous execution of NBAPredictor and find the instance that resulted in the best accuracy

        Returns
        -------
        (highest_val, data): tuple
            A tuple of the best accuracy ever achieved in NBAPredictor along with a dictionary of attributes used in
            that best performing instance
        """
        highest_val = float('-inf')
        data = None
        for instance in self.stats:
            if self.stats[instance]["Accuracy"] > highest_val:
                highest_val = self.stats[instance]["Accuracy"]
                data = self.stats[instance]
        return highest_val, data

    def get_n_best_instances(self, n: int = 5, log=True):
        best_vals = [float('-inf') for x in range(n)]
        best_data = [None for x in range(n)]
        for instance in self.stats:
            acc = self.stats[instance]["best_performer"]["Accuracy"]
            for i, val in enumerate(best_vals):
                if acc > val:
                    best_vals[i] = acc
                    best_data[i] = self.stats[instance]
                    break
        if log:
            self.logger.info(f"The {n} best instances so far in NBAPredictor: ")
            for i, data in enumerate(best_data):
                self.logger.info(f"{i + 1}. Accuracy: {data['best_performer']['Accuracy']}, ")
                self.logger.info(f"Features used: {data['best_performer']['Labels']}")
                self.logger.info(f"Neural Network Topology: {data['nn_shape']}\n")

        return best_data

    def log_best_performer(self) -> None:
        """
        Log data such as features used, accuracies, precision, weights and biases, of the best preforming instance in
        NBAPredictor

        Returns
        -------
        None
        """
        best = self.get_highest_accuracy()
        self.logger.info(f"\n\nThe model with the highest accuracy {best[0]} has the following characteristics: \n")
        for k, v in best[1].items():
            if k != 'best_performer':
                self.logger.info(f"{k} : {v}")
            else:
                self.logger.info(f"Best Accuracy: {v['Accuracy']}")
                self.logger.info("Features used: ")
                for f in v['Labels']:
                    self.logger.info(f)
                for nw, w in v['Vars'].items():
                    self.logger.info(f"{nw}: {w}")

    def get_highest_precision(self) -> Tuple[float, Dict[str, Any]]:
        """
        Loop through each previous execution of NBAPredictor and find the instance that resulted in the best precision

        Returns
        -------
        (highest_val, data): tuple
            A tuple of the best precision ever achieved in NBAPredictor along with a dictionary of attributes used in
            that instance
        """
        highest_val = float('-inf')
        data = None
        for instance in self.stats:
            if self.stats[instance]["Precision"] > highest_val:
                highest_val = self.stats[instance]["Precision"]
                data = self.stats[instance]
        return highest_val, data

    def extract_biases(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates and returns a  dictionary of the biases used at each neuron in the neural network for a previous
        execution instance of NBAPredictor

        Parameters
        ----------
        data: dict
            The dictionary containing data from the previous execution instance of NBAPredictor

        Returns
        -------
        output: dict
            A dictionary of bias names to values used in that instance
        """
        output = dict()
        vars = data["best_performer"]["Vars"]
        for var in vars:
            if re.match(r'dnn/hiddenlayer_[0-9]/bias', var) and re.search("ProximalAdagrad", var) == None:
                layer = re.compile(r'hiddenlayer_\d+(?:\.\d+)?').findall(var)[0]
                output[layer] = vars[var]
        return output

    def extract_weights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates and returns a dictionary of every feature used in some previous execution of NBAPredictor and what
        the initial weights were at each neuron along with the average at each neuron.

        Parameters
        ----------
        data: dict
            The dictionary containing data from the previous execution instance of NBAPredictor

        Returns
        -------
        output: dict
            A dictionary of feature names and the initial weights used for that feature at each neuron (along with
            the average across all) for a previous execution instance of NBAPredictor
        """
        output = dict()
        vars = data["best_performer"]["Vars"]
        labels = data["best_performer"]["Labels"]
        for var in vars:
            if re.match(r'dnn/hiddenlayer_0/kernel', var) and re.search("ProximalAdagrad", var) == None:
                layer_data = vars[var]
                assert len(layer_data) == len(labels), f"Mismatched lengths of labels {len(labels)} and output " \
                                                       f"weights " \
                                                       f"{len(layer_data)}!"
                for i, weight_data in enumerate(layer_data):
                    output[labels[i]] = dict()
                    output[labels[i]]["weights"] = weight_data
                    output[labels[i]]["avg"] = sum(weight_data) / len(weight_data)
        return output

    def write_avg_feature_weight(self) -> Dict[str, float]:
        """
        Updates the 'features.json' file by calculating the average initial weight value for each feature ever used
        in previous NBAPredictor executions.

        Returns
        -------
        output: dict
            A dictionary of feature names along with the average feature weight value across every single execution
            instance of NBAPredictor
        """
        output = dict()
        weights = [self.extract_weights(self.stats[data]) for data in self.stats]
        for i, instance in enumerate(weights):
            for feature in instance:
                output.setdefault(feature, 0.0)
                output[feature] = (weights[i][feature]["avg"] + output[feature]) / 2
        output["Average"] = sum(output.values()) / len(output.values())
        with open(self.feature_file, 'w') as json_file:
            json.dump(output, json_file)
        return output

    def get_best_features(self) -> List[Tuple[str, float]]:
        """
        Creates a list of sorted tuples of format ( <Feature Name> , Average Weight Value) sorted from highest average
        feature weight to lowest and returns it.

        Returns
        -------
        list
            A list of sorted tuples holding the best preforming features in order of positive weight value
        """
        return [(key, self.features[key]) for key in sorted(self.features, key=self.features.get, reverse=True)]
