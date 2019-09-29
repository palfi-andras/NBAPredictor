import configparser
import logging
import os

from tap import Tap

from automated_selection import AutomatedSelection
from league import load_league
from nba_json_parser import NBAJsonParser
from read_stats import ReadStats
from tensorflow_operations import TensorflowOperations


class NBAPredictorArguments(Tap):
    dir: str = './resources/nba'  # Optional path for directory of where JSON
    # files are located when scraped from basketballrefernce.com
    rebuild: bool = False  # Pass this flag in to rebuild the library of players and previous games. Whenever the   #
    # library is built, it is saved so it can be reused again for the next execution of the program. This flag forces
    # the rebuild of the entire League and will add some time to program startup
    league_save: str = './resources/league.pkl'  # Location of the league save file (Pickle Object)
    config_file: str = './resources/config.ini'  # Location of the NBAPredictor Config File


class ParsedConfigs:

    def __init__(self, path: str):
        self.configs = configparser.ConfigParser()
        self.configs.read(path)
        self.randomize_nn_shape = bool(self.configs["DEFAULT"]["RANDOMIZE_NN_SHAPE"])
        if self.randomize_nn_shape == "False":
            self.nn_shape = None
        else:
            self.nn_shape = [int(x) for x in self.configs["DEFAULT"]["NN_SHAPE"].split()]
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
        if self.configs["DEFAULT"]["NORMALIZE_WEIGHTS_ACCORDING_TO_RECORD"] == "True":
            self.normalize_weights = True
        else:
            self.normalize_weights = False


if __name__ == '__main__':
    args: NBAPredictorArguments = NBAPredictorArguments().parse_args()
    if not args.rebuild and not os.path.isfile(args.league_save):
        args.rebuild = True
    if args.rebuild:
        league = NBAJsonParser(args.dir).generate_league_object()
        league.save_league(args.league_save)
    else:
        league = load_league(args.league_save)
    parsed_configs = ParsedConfigs(args.config_file)
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
        read_stats = ReadStats(parsed_configs.stat_location, parsed_configs.features_location, logger=logger)
        read_stats.log_best_performer()
        exit(0)
    for x in range(0, run_size):
        read_stats = ReadStats(parsed_configs.stat_location, parsed_configs.features_location, logger=logger)
        selector = AutomatedSelection(read_stats, strategy=parsed_configs.feature_selection_strategy,
                                      nn_shape=parsed_configs.nn_shape)
        tfops = TensorflowOperations(league=league, num_epochs=parsed_configs.epochs,
                                     learning_rate=parsed_configs.learning_rate, nn_shape=selector.nn_shape,
                                     season=parsed_configs.season, split=parsed_configs.train_size,
                                     outfile=parsed_configs.stat_location, model_dir=f"{parsed_configs.model_dir}/"
                                                                                     f"{selector.model_name}",
                                     features=selector.features, logger=logger, mode=parsed_configs.method,
                                     normalize_weights=parsed_configs.normalize_weights)
        tfops.run()
