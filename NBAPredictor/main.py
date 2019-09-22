import configparser
import os

from tap import Tap

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
        self.nn_shape = self.extract_model_shape()
        self.epochs = int(self.configs["DEFAULT"]["EPOCHS"])
        self.learning_rate = float(self.configs["DEFAULT"]["LEARNING_RATE"])
        self.train_size = float(self.configs["DEFAULT"]["TRAIN_SIZE"])
        self.season = self.configs["DEFAULT"]["SEASON"]
        self.stat_location = self.configs["DEFAULT"]["STAT_LOCATION"]
        self.model_dir = self.configs["DEFAULT"]["MODEL_DIR"]
        self.features_location = self.configs["DEFAULT"]["FEATURE_PERFORMANCE_LOCATION"]

    def extract_model_shape(self):
        shape = self.configs["DEFAULT"]["NN_SHAPE"]
        s = shape.split()
        return [int(x) for x in s]


if __name__ == '__main__':
    args: NBAPredictorArguments = NBAPredictorArguments().parse_args()
    parsed_configs = ParsedConfigs(args.config_file)
    read_stats = ReadStats(parsed_configs.stat_location, parsed_configs.features_location)
    if not args.rebuild and not os.path.isfile(args.league_save):
        args.rebuild = True
    if args.rebuild:
        league = NBAJsonParser(args.dir).generate_league_object()
        league.save_league(args.league_save)
    else:
        league = load_league(args.league_save)
    tfops = TensorflowOperations(league=league, num_epochs=parsed_configs.epochs,
                                 learning_rate=parsed_configs.learning_rate, nn_shape=parsed_configs.nn_shape,
                                 season=parsed_configs.season, split=parsed_configs.train_size,
                                 outfile=parsed_configs.stat_location, model_dir=parsed_configs.model_dir)
    tfops.run_neural_network()
