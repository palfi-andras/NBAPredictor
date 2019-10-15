#!/usr/bin/env python3
import configparser
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "analyze"))
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))
sys.path.append(os.path.join(os.path.dirname(__file__), "training"))

from automated_selection import AutomatedSelection
from league import load_league
from nba_json_parser import NBAJsonParser
from read_stats import ReadStats
from tensorflow_operations import TensorflowOperations


class ParsedConfigs:

    def __init__(self, path: str):
        self.configs = configparser.ConfigParser()
        self.configs.read(path)
        self.rebuild = True if self.configs["DEFAULT"]["REBUILD"] == "True" else False
        self.league_save = self.configs["DEFAULT"]["LEAGUE_PICKLE_OBJECT"]
        self.nba_json_dir = self.configs["DEFAULT"]["NBA_JSON_GAME_PATH"]
        self.randomize_nn_shape = True if self.configs["DEFAULT"]["RANDOMIZE_NN_SHAPE"] == "True" else False

        self.nn_shape = [int(x) for x in
                         self.configs["DEFAULT"]["NN_SHAPE"].split()] if not self.randomize_nn_shape else None
        self.method = self.configs["DEFAULT"]["METHOD"]
        assert self.method in ["DNN", "SVM"]
        self.epochs = int(self.configs["DEFAULT"]["EPOCHS"])
        self.learning_rate = float(self.configs["DEFAULT"]["LEARNING_RATE"])
        self.train_size = float(self.configs["DEFAULT"]["TRAIN_SIZE"])
        self.season = self.configs["DEFAULT"]["SEASON"]
        self.stat_location = self.configs["DEFAULT"]["STAT_LOCATION"]
        self.model_dir = self.configs["DEFAULT"]["MODEL_DIR"]
        self.features_location = self.configs["DEFAULT"]["FEATURE_PERFORMANCE_LOCATION"]
        self.feature_selection_strategy = self.configs["DEFAULT"]["FEATURE_SELECTION_STRATEGY"]
        self.batch_run = int(self.configs["DEFAULT"]["BATCH_RUN"])
        self.batch_feature_selection_strategy = self.configs["DEFAULT"]["BATCH_RUN_FEATURE_SELECTION_STRATEGY"]
        if "UseNBestAndRandom" in self.feature_selection_strategy:
            self.feature_selection_strategy = self.configs["DEFAULT"]["FEATURE_SELECTION_STRATEGY"].split()[0]
            self.n_best = self.configs["DEFAULT"]["FEATURE_SELECTION_STRATEGY"].split()[1]
        if "UseNBestAndRandom" in self.batch_feature_selection_strategy:
            self.batch_feature_selection_strategy = \
                self.configs["DEFAULT"]["BATCH_RUN_FEATURE_SELECTION_STRATEGY"].split()[0]
            self.n_best = self.configs["DEFAULT"]["BATCH_RUN_FEATURE_SELECTION_STRATEGY"].split()[1]
        self.log_file = self.configs["DEFAULT"]["LOG_FILE"]
        self.mode = self.configs["DEFAULT"]["MODE"]
        self.normalize_weights = True if self.configs["DEFAULT"][
                                             "NORMALIZE_WEIGHTS_ACCORDING_TO_RECORD"] == "True" else False
        self.cache = True if self.configs["DEFAULT"]["CACHE"] == "True" else False
        self.cache_dir = self.configs["DEFAULT"]["NUMPY_CACHED_DATA_DIR"]
        self.predict_next_season = True if self.configs["DEFAULT"]["PREDICT_NEXT_SEASON"] == "True" else False
        self.next_season_schedule = self.configs["DEFAULT"]["NEXT_SEASON_SCHEDULE"]
        self.feature_selection_strategy = "Predict" if self.predict_next_season else self.feature_selection_strategy


if __name__ == '__main__':
    config_file = './resources/config.ini'
    parsed_configs = ParsedConfigs(config_file)
    if not parsed_configs.rebuild and not os.path.isfile(parsed_configs.league_save):
        parsed_configs.rebuild = True
    if parsed_configs.rebuild:
        league = NBAJsonParser(parsed_configs.nba_json_dir).generate_league_object()
        league.save_league(parsed_configs.league_save)
    else:
        league = load_league(parsed_configs.league_save)
    if parsed_configs.batch_run <= 0:
        run_size = 1
        strategy = parsed_configs.feature_selection_strategy
    else:
        run_size = parsed_configs.batch_run
        strategy = parsed_configs.batch_feature_selection_strategy
    logger = logging.getLogger("NBAPredictor")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(parsed_configs.log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    if parsed_configs.mode == "Analyze":
        read_stats = ReadStats(parsed_configs.stat_location, parsed_configs.features_location)
        read_stats.get_n_best_instances()
        exit(0)
    for x in range(0, run_size):
        read_stats = ReadStats(parsed_configs.stat_location, parsed_configs.features_location)
        selector = AutomatedSelection(read_stats, strategy=parsed_configs.feature_selection_strategy,
                                      nn_shape=parsed_configs.nn_shape)
        model_name = f"{parsed_configs.model_dir}/{selector.model_name}" if not parsed_configs.normalize_weights else\
            f"{parsed_configs.model_dir}/{selector.model_name}_normalized"
        tfops = TensorflowOperations(league=league, num_epochs=parsed_configs.epochs,
                                     learning_rate=parsed_configs.learning_rate, nn_shape=selector.nn_shape,
                                     season=parsed_configs.season, split=parsed_configs.train_size,
                                     outfile=parsed_configs.stat_location, model_dir=model_name,
                                     features=selector.features, mode=parsed_configs.method,
                                     normalize_weights=parsed_configs.normalize_weights,
                                     cache_numpy_structures=parsed_configs.cache, cache_dir=parsed_configs.cache_dir,
                                     predict_next_season=parsed_configs.predict_next_season,
                                     next_season_csv=parsed_configs.next_season_schedule)
        tfops.run()
