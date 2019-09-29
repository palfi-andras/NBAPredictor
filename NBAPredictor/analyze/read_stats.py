import json
import os
import re
from collections import OrderedDict
from typing import Tuple, Dict, Any
import logging

"""
A class to read the predictions.json file that stores data about the best results from a previous run of NBAPredictor.
Then the features used can be examined and their corresponding weights to see which ones are doing more/less.
"""


# TODO THIS IS BROKEN WITH SVM COMPATIBILITY
class ReadStats:

    def __init__(self, stats_file: str, feature_file: str, logger: logging):
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

    def get_last_used_features(self):
        """
        Returns the set of features used in the last run
        """
        return self.stats[next(reversed(OrderedDict(self.stats)))]['best_performer']['Labels']

    def get_highest_accuracy(self) -> Tuple[float, Dict[str, Any]]:
        """
        Returns the instance with the best accuracy
        :return:
        """
        highest_val = float('-inf')
        data = None
        for instance in self.stats:
            if self.stats[instance]["Accuracy"] > highest_val:
                highest_val = self.stats[instance]["Accuracy"]
                data = self.stats[instance]
        return highest_val, data

    def log_best_performer(self):
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
        Returns the instance with the best precision
        :return:
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
        Returns a dictionary of the biases at each neuron in the neural network.
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
        Returns a dictionary of the weights used for each neuron and the for every feature considered in input along
        with
        average values
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

    def write_avg_feature_weight(self):
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

    def get_best_features(self):
        return [(key, self.features[key]) for key in sorted(self.features, key=self.features.get, reverse=True)]
