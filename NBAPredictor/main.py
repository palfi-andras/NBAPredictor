import configparser
import os

from tap import Tap

from feature_selection import FeatureSelection
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
        self.nn_shape = [int(x) for x in self.configs["DEFAULT"]["NN_SHAPE"].split()]
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
    if parsed_configs.batch_run >= 0:
        run_size = 1
        strategy = parsed_configs.feature_selection_strategy
    else:
        run_size = parsed_configs.batch_run
        strategy = parsed_configs.batch_feature_selection_strategy
    for x in range(0, run_size):
        read_stats = ReadStats(parsed_configs.stat_location, parsed_configs.features_location)
        feature_selector = FeatureSelection(read_stats, strategy=parsed_configs.feature_selection_strategy)
        print(f"Experiment #{x + 1}. Running DNN on the {parsed_configs.season}NBA Season for {parsed_configs.epochs} "
              f"epochs with the following NN shape: {parsed_configs.nn_shape} and the following input features: "
              f"{feature_selector.features}")

        tfops = TensorflowOperations(league=league, num_epochs=parsed_configs.epochs,
                                     learning_rate=parsed_configs.learning_rate, nn_shape=parsed_configs.nn_shape,
                                     season=parsed_configs.season, split=parsed_configs.train_size,
                                     outfile=parsed_configs.stat_location, model_dir=parsed_configs.model_dir,
                                     features=feature_selector.features)
        tfops.run_neural_network()
