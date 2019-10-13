import hashlib
import logging
import os
import pickle
import re
import time
from statistics import stdev
from typing import List, Tuple, Dict, Union

import numpy as np

from game import Game
from game_period import GamePeriod
from league import League
from player import Player
from player_stat_types import PlayerStatTypes
from team import Team

# TODO all of these utility functions should probably be brought into a utility class. This class is messy enough as
# it is and we dont need the extra clutter


POSSIBLE_FEATURES = ["ReboundSpread", "OffensiveReboundSpread", "DefensiveReboundSpread", "AssistSpread",
                     "TurnoverSpread", "FieldGoalPercentSpread", "ThreePointPercentSpread", "FreeThrowPercentSpread",
                     "FieldGoalsAttemptedSpread", "ThreePointsAttemptedSpread", "FreeThrowsAttemptedSpread",
                     "BestPlayerSpread", "TeamRecordSpread", "StealSpread", "BlockSpread", "PersonalFoulSpread",
                     "TrueShootingPercentSpread", "ThreePointRateSpread", "FreeThrowRateSpread",
                     "OffensiveRatingSpread", "DefensiveRatingSpread", "AssistToTurnoverSpread",
                     "StealToTurnoverSpread", "HOBSpread", "ExperienceSpread", "HomeFieldAdvantage"]

DEFAULT_FEATURES = ["ReboundSpread", "OffensiveReboundSpread", "DefensiveReboundSpread", "AssistSpread",
                    "TurnoverSpread", "FieldGoalPercentSpread", "ThreePointPercentSpread", "FreeThrowPercentSpread",
                    "FieldGoalsAttemptedSpread", "ThreePointsAttemptedSpread", "FreeThrowsAttemptedSpread",
                    "BestPlayerSpread", "HomeFieldAdvantage"]


def determine_best_player_from_team(team: Team) -> Player:
    """
    Determines the best player from a given team for an NBAMatch

    Parameters
    ----------
    team: Team
        The team the player is on

    Returns
    -------
    Player
        The player with the highest FIC score
    """
    best_val = float('-inf')
    best_player = None
    for player in team.players:
        fic = player.stats.get(PlayerStatTypes.FIC)
        if fic is not None:
            if fic > best_val:
                best_player = player
    return best_player


def center_values(values: List[float]) -> List[float]:
    """
    Centers a list of values by subtracting the average and dividing by the stdev for all values in the list

    Parameters
    ----------
    values: list
        The list of values to center

    Returns
    -------
    list
        The same list but with all values centered
    """
    try:
        avg = sum(values) / len(values)
        return [(x - avg) / stdev(values) for x in values]
    except ZeroDivisionError:
        return values


def center_numpy_features(values: np.array, normalize=True) -> np.array:
    """
    Centers a numpy array by subtracting the average and dividing by the stdev for all values in the list

    Parameters
    ----------
    values: NumPy Array
        The array of values
    normalize: bool
        Whether to normalize the values in the NumPy Array or not.

    Returns
    -------
    The same NumPy array but with centered values
    """
    try:
        with np.nditer(values, op_flags=['readwrite']) as it:
            for i, x in enumerate(it):
                current_col = i % values.shape[1]
                if not normalize:
                    col_vals = values[:, current_col]
                    col_avg = np.sum(col_vals) / col_vals.size
                    col_std_dev = stdev(col_vals.tolist())
                    x[...] = (x - col_avg) / col_std_dev
                else:
                    if (values.shape[1] - 1) != current_col:
                        col_vals = values[:, current_col]
                        col_avg = np.sum(col_vals) / col_vals.size
                        col_std_dev = stdev(col_vals.tolist())
                        x[...] = (x - col_avg) / col_std_dev
        return values
    except ZeroDivisionError:
        return values


def check_for_multiple_seasons(seasons: str) -> List[str]:
    """
    This utility function is used for when a user selects more than one NBA season to parse, for example 2010-2018.
    It will break apart the string into the individual year string (2010-2011, 2011-2012, etc) and then return them
    as a list.

    Parameters
    ----------
    seasons: str
        The selected season(s) to parse

    Returns
    -------
    list
        A list individual NBA seasons to use as input data
    """
    proper_syntax = r'20[0-9][0-9]-20[0-9][0-9]'
    assert re.match(proper_syntax, seasons), f"{seasons} is not a recognized string in the from of " \
                                             f"{proper_syntax}!"
    start_year = int(seasons[2:4])
    end_year = int(seasons[-2:])
    assert end_year > start_year, f"The end year {end_year} occurs before the start year {start_year}!"
    season_list = list()
    if (end_year - start_year) > 1:
        num_years = end_year - start_year
        for x in range(start_year, start_year + num_years):
            year = x if x > 9 else f"0{x}"
            next_year = x + 1 if x + 1 > 9 else f"0{x + 1}"
            season_list.append(f"20{year}-20{next_year}")
    else:
        season_list.append(seasons)
    return season_list


class ReadGames:
    """
    The ReadGames class is an extremely useful class within NBAPredictor that is tasked with preparing both datasets
    for training and testing in an abstract enough way so that they can be used with multiple ML approaches (DNN and
    SVM). This class preprares NumPy Arrays given a subset of data requested from the main League object. It will
    extract all the features requested and includes option for weight normalizations.

    Attributes
    ----------
    league: League
        The main League object created by the NbaJsonParser that contains ALL historical game data.
    logger: logging
        A logger for this class
    features: list
        A list of features that will be used for this dataset. When NBAPredictor starts, the features passed in here
        by the AutomatedSelector will be the only data points extracted from the main League object in order to
        reduce memory footprint
    normalize_weights: bool
        NBAPredictor has an ability to downplay/strengthen certain training or testing examples. It does this by
        determining how "different" the two teams are. If there is a large talent dispairty, there is a high chance
        the result of the outcome was because of this talent disparity, not because of any in-game features.
        Therefore, if the flag is passed, games between teams with a large talent disparity will be "downplayed" as
        noteworthy examples for training/testing and conversely games between equal caliber teams will be strengthened.
    training_size: int
        The amount of games to use for the training subset
    sorted_games: list
        A sorted list of games from the subset of seasons requested. The games are listed in chronological order of
        date played.
    training_features: dict
        A dictionary of every feature requested, along with a NumPy Array of every value for that feature for each
        game in the subset of training games. The array is equal to training_size length.
    training_labels: list
        A list of labels indicating the actual outcome of each game in the subset of training games. Valid options
        are 'H' for home team win or 'A' for away team win. The array is equal to training_size length.
    testing_features: dict
        A dictionary of every feature requested, along with a NumPy Array of every value for that feature for each
        game in the subset of testing games
    testing_labels: list
        A list of labels indicating the actual outcome of each game in the subset of testing games. Valid options
        are 'H' for home team win or 'A' for away team win

    Methods
    -------
    parse_whole_season
        Parses the entire subset of seasons required when this object was intiialized. This will divide that subset
        of games at the split requested during initialization and extract the features requested. If
        normalize_weights is True, it will also created the weight column for each training and testing example
    parse_whole_season_svm_format
        Exact same behavior as parse_whole_season, but outputs slightly different data structures for compatibility
        reasons with the SVM model.
    map_feature_name_to_actual_value (name: str, game: Game)
        Takes a feature name and calculates that feature value from the given game. This method essentially serves as
        a wrapper and calls the other internal functions of this class in order to extract the right features from
        each game
    get_winner (game: Game)
        Determines the winner of a given game
    determine_best_player_spread (game: Game)
        Determines the spread between the home players best players FIC score and the away best players FIC score
    determine_home_field_advantage_spread (game: Game)
        Calculates the percent of games the home team usually wins subtracted by the percent of games the away team
        usually wins on the road. So a positive weight means that the home team has a higher chance to win based off
        their location and a negative value means that the home field advantage feature does not matter as the away
        team is better
    determine_experience_spread (game: Game)
        Determines the spread between the home teams players Experience levels and the away teams players Experience
        levels
    get_team_record_differential (game: Game)
        Returns the teams record differential at this point of the season (before this game) as a float of games won
        against all games played. This method may be used to normalize the effects of the other features. This may be
        desirable in instances where the disparity in record between the teams is large so the effects of other in
        game features are lowered. Higher values returned by this function mean that this game is a more noteworth
        example
    determine_rebound_differential(game: Game)
        Determines the difference for rebounds between the two teams.
    determine_offensive_rebound_differential (game: Game)
        Determines the difference for offensive rebounds between the two teams.
    determine_defensive_rebound_differential (game: Game)
        Determines the difference for defensive rebounds between the two teams.
    determine_assist_spread (game: Game)
        Determines the difference for assists between the two teams.
    determine_turnover_spread (game: Game)
        Determines the difference for turnovers between the two teams.
    determine_field_goal_percent_spread (game: Game)
        Determines the difference for field goal percentage between the two teams.
    determine_three_point_percent_spread (game: Game)
        Determines the difference for three point percentage between the two teams.
    determine_free_throw_percent_spread (game: Game)
        Determines the difference for free throw percentage between the two teams.
    determine_field_goals_attempted_spread (game: Game)
        Determines the difference for field goals attempted between the two teams.
    determine_three_points_attempted_spread (game: Game)
        Determines the difference for three points attempted between the two teams.
    determine_free_throws_attempted_spread (game: Game)
        Determines the difference for free throws attempted between the two teams.
    determine_steals_spread (game: Game)
        Determines the difference for steals between the two teams.
    determine_blocks_spread (game: Game)
        Determines the difference for blocks between the two teams.
    determine_personal_foulds_spread (game: Game)
        Determines the difference for personal fouls between the two teams.
    determine_true_shooting_percent_spread (game: Game)
        Determines the difference for true shooting percentage between the two teams.
    determine_three_point_rate_spread (game: Game)
        Determines the difference for three point rate between the two teams.
    determine_free_throw_rate_spread (game: Game)
        Determines the difference for free throw rate between the two teams.
    determine_offensive_rating_spread (game: Game)
        Determines the difference for offensive rating between the two teams.
    determine_defensive_rating_spread (game: Game)
        Determines the difference for defensive rating between the two teams.
    determine_assist_to_turnover_spread (game: Game)
        Determines the difference for assist-to-turnover-ratio between the two teams.
    determine_steal_to_turnover_spread (game: Game)
        Determines the difference for steal-to-turnover-ratio between the two teams.
    determine_hob_spread_spread (game: Game)
        Determines the difference for hands on ball between the two teams.
    """

    def __init__(self, leauge: League, season: str, split: float, cache_dir: str,
            features: List[str] = DEFAULT_FEATURES, svm_compat=False, normalize_weights=False, cache=False,
            initialize=True):
        """
        Parameters
        ----------
        league: League
            The main League object created by the NbaJsonParser that contains ALL historical game data.
        season: str
            The season, or seasons that should be used for training/testing. Valid options include single seasons
            such as
            2010-2011 or ranges such as 2010-2018
        split: float
            The split at which to divide all of the games from the seasons requested at for training/testing. Bounded
            between 0.01 and 0.99
        logger: logging
            A logger for this class
        features: list
            A list of features that will be used for this dataset. When NBAPredictor starts, the features passed in here
            by the AutomatedSelector will be the only data points extracted from the main League object in order to
            reduce memory footprint
        svm_compat: bool
            The SVM learning approach within NBAPredictor expects a slightly different format for input data,
            so pass this flag if the program is being run with the SVM mode.
        normalize_weights: bool
            NBAPredictor has an ability to downplay/strengthen certain training or testing examples. It does this by
            determining how "different" the two teams are. If there is a large talent dispairty, there is a high chance
            the result of the outcome was because of this talent disparity, not because of any in-game features.
            Therefore, if the flag is passed, games between teams with a large talent disparity will be "downplayed" as
            noteworthy examples for training/testing and conversely games between equal caliber teams will be
            strengthened.
        """
        self.logger = logging.getLogger(f"NBAPredictor.{self.__class__.__name__}")
        start_time = time.time()
        self.logger.info("Loading data set...")
        self.leauge = leauge
        self.normalize_weights = normalize_weights
        if self.normalize_weights:
            self.logger.info(f"Normalization is turned on")
        else:
            self.logger.info(f"Normalization is turned off")
        self.seasons = [s for i, s in enumerate(check_for_multiple_seasons(season)) if s in self.leauge.seasons_dict]
        self.logger.info(f"Using the following NBA seasons: {self.seasons}")
        assert all(e in POSSIBLE_FEATURES for e in features)
        assert (0.01 < split < 0.99)
        self.features = features

        if len(self.seasons) == 1:
            self.training_size = round(len(self.leauge.seasons_dict[season]) * split)
            self.sorted_games = sorted(self.leauge.seasons_dict[season].__iter__(),
                                       key=lambda x: re.sub(r"[A-Z]", "", x.code))
        else:
            self.training_size = round(sum(len(self.leauge.seasons_dict[s]) for s in self.seasons) * split)
            self.sorted_games = list()
            for s in self.seasons:
                self.sorted_games.extend(
                    sorted(self.leauge.seasons_dict[s].__iter__(), key=lambda x: re.sub(r"[A-Z]", "", x.code)))

        if initialize:
            self.logger.info(f"Size of training set: {self.training_size}, size of testing set: "
                             f"{len(self.sorted_games) - self.training_size} ({split * 100}% split)")
            if not svm_compat:
                if cache:
                    if self.load_instance(cache_dir):
                        self.training_features, self.training_labels, self.testing_features, self.testing_labels = \
                            self.load_instance(
                            cache_dir)
                    else:
                        self.training_features, self.training_labels, self.testing_features, self.testing_labels = \
                            self.parse_whole_season()
                        self.cache_instance(cache_dir)
                else:
                    self.training_features, self.training_labels, self.testing_features, self.testing_labels = \
                        self.parse_whole_season()
            else:
                self.training_features, self.training_labels, self.testing_features, self.testing_labels = \
                    self.parse_whole_season_svm_format()
            self.logger.info(f"Prepared all data sets in {time.time() - start_time} seconds")

    def cache_instance(self, dir: str) -> None:
        """
        Writes this instance out to a pickle file so that if a similair instance is started with the same features,
        we dont have to do all the parsing and normalization again. Note this will only load if you run the same
        exact features, games, and training split.

        Parameters
        ----------
        dir: str
            Path to a directory where pickle files should be dumped

        Returns
        -------
        None
        """
        assert self.seasons and self.features and self.training_size and self.sorted_games
        name = f"instance-{self.training_size}-{len(self.sorted_games)}-" \
               f"{str(hashlib.sha1(str(self.seasons).encode('utf-8')).hexdigest())}-" \
               f"{str(hashlib.sha1(str(self.features).encode('utf-8')).hexdigest())}"
        path = os.path.join(dir, name)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def load_instance(self, dir: str) -> Union[Tuple, None]:
        """
        Attempts to load a cached instance of this class and extract the already configured NUMPY structures,
        so that they do not have to be generated again.

        Parameters
        ----------
        dir: str
            Path to the directory containing instance files

        Returns
        -------
        Tuples of training/testing features and labels, or nothing if an instance cant be found0
        """
        assert self.seasons and self.features and self.training_size and self.sorted_games
        name = f"instance-{self.training_size}-{len(self.sorted_games)}-" \
               f"{str(hashlib.sha1(str(self.seasons).encode('utf-8')).hexdigest())}-" \
               f"{str(hashlib.sha1(str(self.features).encode('utf-8')).hexdigest())}"
        path = os.path.join(dir, name)
        if os.path.isfile(path):
            with open(path, 'rb') as pickle_file:
                instance: ReadGames = pickle.load(pickle_file)
                return instance.training_features, instance.training_labels, instance.testing_features, \
                       instance.testing_labels
        return None

    def parse_whole_season(self) -> Tuple[Dict[str, np.array], np.array, Dict[str, np.array], np.array]:
        """
        Parses the entire subset of seasons required when this object was initialized. This will divide that subset
        of games at the split requested during initialization and extract the features requested. If
        normalize_weights is True, it will also created the weight column for each training and testing example

        Returns
        -------
        training_features: dict
            A dictionary of every feature requested, along with a NumPy Array of every value for that feature for each
            game in the subset of training games. The array is equal to training_size length.
        training_lables: NumPy Array
            A list of labels indicating the actual outcome of each game in the subset of training games. Valid options
            are 'H' for home team win or 'A' for away team win. The array is equal to training_size length.
        testing_features: dict
            A dictionary of every feature requested, along with a NumPy Array of every value for that feature for each
            game in the subset of testing games
        testing_labels: NumPy Array
            A list of labels indicating the actual outcome of each game in the subset of testing games. Valid options
            are 'H' for home team win or 'A' for away team win
        """
        training_features = dict()
        training_labels = list()
        testing_features = dict()
        testing_labels = list()
        for label in self.features:
            training_features.setdefault(label, list())
            testing_features.setdefault(label, list())
        for x in range(0, self.training_size):
            game = self.sorted_games[x]
            self.logger.info(f"Extracting requested features for game {game.code}")
            training_labels.append(self.get_winner(game))
            for feature in self.features:
                training_features[feature].append(self.map_feature_name_to_actual_value(feature, game))
            if self.normalize_weights:
                training_features.setdefault("weight", list())
                normalized_game_value = self.map_feature_name_to_actual_value("TeamRecordSpread", game)
                self.logger.info(f"Calculated the normalized game value to be {normalized_game_value}.")
                training_features["weight"].append(normalized_game_value)
        for x in range(self.training_size + 1, len(self.sorted_games)):
            game = self.sorted_games[x]
            self.logger.info(f"Extracting requested features for game {game.code}")
            testing_labels.append(self.get_winner(game))
            for feature in self.features:
                testing_features[feature].append(self.map_feature_name_to_actual_value(feature, game))
            if self.normalize_weights:
                testing_features.setdefault("weight", list())
                normalized_game_value = self.map_feature_name_to_actual_value("TeamRecordSpread", game)
                self.logger.info(f"Calculated the normalized game value to be {normalized_game_value}.")
                testing_features["weight"].append(normalized_game_value)
        for item in training_features:
            self.logger.info(f"Centering feature set {item} for training data...")
            training_features[item] = center_values(training_features[item])
            training_features[item] = np.array(training_features[item])
        for item in testing_features:
            self.logger.info(f"Centering feature set {item} for testing data...")
            testing_features[item] = center_values(testing_features[item])
            testing_features[item] = np.array(testing_features[item])
        training_labels = np.array([label for label in training_labels])
        testing_labels = np.array([label for label in testing_labels])
        self.logger.info(f"Number of games used for training: {len(training_labels)}")
        self.logger.info(f"Number of games used for testing: {len(testing_labels)}")
        return training_features, training_labels, testing_features, testing_labels

    def parse_whole_season_svm_format(self):
        """
        Exact same behavior as parse_whole_season, but outputs slightly different data structures for compatibility
        reasons with the SVM model.

        Returns
        -------
        training_features: dict
            A dictionary of every feature requested, along with a NumPy Array of every value for that feature for each
            game in the subset of training games. The array is equal to training_size length.
        training_lables: NumPy Array
            A list of labels indicating the actual outcome of each game in the subset of training games. Valid options
            are 'H' for home team win or 'A' for away team win. The array is equal to training_size length.
        testing_features: dict
            A dictionary of every feature requested, along with a NumPy Array of every value for that feature for each
            game in the subset of testing games
        testing_labels: NumPy Array
            A list of labels indicating the actual outcome of each game in the subset of testing games. Valid options
            are 'H' for home team win or 'A' for away team win
        """
        training_features = None
        training_labels = np.array([])
        testing_features = None
        testing_labels = np.array([])
        for x in range(0, self.training_size):
            game = self.sorted_games[x]
            training_labels = np.append(training_labels, [self.get_winner(game)])
            game_array = np.array([])
            for feature in self.features:
                game_array = np.append(game_array, [self.map_feature_name_to_actual_value(feature, game)])
            if self.normalize_weights:
                game_array = np.append(game_array, [self.map_feature_name_to_actual_value("TeamRecordSpread", game)])
            if training_features is None:
                training_features = game_array
            else:
                training_features = np.vstack((training_features, game_array))
        for x in range(self.training_size + 1, len(self.sorted_games)):
            game = self.sorted_games[x]
            testing_labels = np.append(testing_labels, [self.get_winner(game)])
            game_array = np.array([])
            for feature in self.features:
                game_array = np.append(game_array, [self.map_feature_name_to_actual_value(feature, game)])
            if self.normalize_weights:
                game_array = np.append(game_array, [self.map_feature_name_to_actual_value("TeamRecordSpread", game)])
            if testing_features is None:
                testing_features = game_array
            else:
                testing_features = np.vstack((testing_features, game_array))
        testing_features = center_numpy_features(testing_features, normalize=self.normalize_weights)
        training_features = center_numpy_features(training_features, normalize=self.normalize_weights)
        self.logger.info(f"Number of games used for training: {len(training_labels)}")
        self.logger.info(f"Number of games used for testing: {len(testing_labels)}")
        return training_features, training_labels, testing_features, testing_labels

    def map_feature_name_to_actual_value(self, name: str, game: Game) -> float:
        """
        Takes a feature name and calculates that feature value from the given game. This method essentially serves as
        a wrapper and calls the other internal functions of this class in order to extract the right features from
        each game

        Parameters
        ----------
        name: str
            The feature name to extract the value for
        game: Game
            The game object to extract the feature from

        Returns
        -------
        float
            The value of the feature for the given game

        Raises
        ------
        RuntimeError
            If name is not a recognized feature name, an exception is raised

        """
        if name == "ReboundSpread":
            return self.determine_rebound_differential(game)
        elif name == "OffensiveReboundSpread":
            return self.determine_offensive_rebound_differential(game)
        elif name == "DefensiveReboundSpread":
            return self.determine_defensive_rebound_differential(game)
        elif name == "AssistSpread":
            return self.determine_assist_spread(game)
        elif name == "TurnoverSpread":
            return self.determine_turnover_spread(game)
        elif name == "FieldGoalPercentSpread":
            return self.determine_field_goal_percent_spread(game)
        elif name == "ThreePointPercentSpread":
            return self.determine_three_point_percent_spread(game)
        elif name == "FreeThrowPercentSpread":
            return self.determine_free_throw_percent_spread(game)
        elif name == "FieldGoalsAttemptedSpread":
            return self.determine_field_goals_attempted_spread(game)
        elif name == "ThreePointsAttemptedSpread":
            return self.determine_three_points_attempted_spread(game)
        elif name == "FreeThrowsAttemptedSpread":
            return self.determine_free_throws_attempted_spread(game)
        elif name == "BestPlayerSpread":
            return self.determine_best_player_spread(game)
        elif name == "TeamRecordSpread":
            return self.get_team_record_differential(game)
        elif name == "StealSpread":
            return self.determine_steals_spread(game)
        elif name == "BlockSpread":
            return self.determine_blocks_spread(game)
        elif name == "PersonalFoulSpread":
            return self.determine_personal_foulds_spread(game)
        elif name == "TrueShootingPercentSpread":
            return self.determine_true_shooting_percent_spread(game)
        elif name == "ThreePointRateSpread":
            return self.determine_three_point_rate_spread(game)
        elif name == "FreeThrowRateSpread":
            return self.determine_free_throw_rate_spread(game)
        elif name == "OffensiveRatingSpread":
            return self.determine_offensive_rating_spread(game)
        elif name == "DefensiveRatingSpread":
            return self.determine_defensive_rating_spread(game)
        elif name == "AssistToTurnoverSpread":
            return self.determine_assist_to_turnover_spread(game)
        elif name == "StealToTurnoverSpread":
            return self.determine_steal_to_turnover_spread(game)
        elif name == "HOBSpread":
            return self.determine_hob_spread_spread(game)
        elif name == "ExperienceSpread":
            return self.determine_experience_spread(game)
        elif name == "HomeFieldAdvantage":
            return self.determine_home_field_advantage_spread(game)
        else:
            raise RuntimeError(f"{name} is not an implemented or recognized feature name!")

    def get_winner(self, game: Game) -> str:
        """
        Determines the winner of a given game

        Parameters
        ----------
        game: Game

        Returns
        -------
        str
            'H' if the home team won, 'A' otherwise.
        """
        home_team_score = game.home_team.scores.get(GamePeriod.TOTAL)
        away_team_score = game.away_team.scores.get(GamePeriod.TOTAL)
        return 'H' if home_team_score > away_team_score else 'A'

    def determine_best_player_spread(self, game: Game) -> float:
        """
        Determines the spread between the home players best players FIC score and the away best players FIC score

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        home_team_best_player = determine_best_player_from_team(game.home_team)
        away_team_best_player = determine_best_player_from_team(game.away_team)
        return home_team_best_player.stats.get(PlayerStatTypes.FIC) - away_team_best_player.stats.get(
            PlayerStatTypes.FIC)

    def determine_home_field_advantage_spread(self, game: Game) -> float:
        """
        Calculates the percent of games the home team usually wins subtracted by the percent of games the away team
        usually wins on the road. So a positive weight means that the home team has a higher chance to win based off
        their location and a negative value means that the home field advantage feature does not matter as the away
        team is better

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
            A float that represents "home field advantage". Higher values indicate better odds for the home team
            winning at home, whereas lower or negative values indicate better odds for the away team.
        """
        home_team_previous_games = [g for g in self.sorted_games if
                                    g.date < game.date and g.season == game.season and g.home_team.name ==
                                    game.home_team.name]
        away_team_previous_games = [g for g in self.sorted_games if
                                    g.date < game.date and g.season == game.season and g.away_team.name ==
                                    game.away_team.name]
        try:
            home_team_win_pct_at_home = len([g for g in home_team_previous_games if
                                             int(g.home_team.scores.get(GamePeriod.TOTAL)) > int(
                                                 g.away_team.scores.get(GamePeriod.TOTAL))]) / len(
                home_team_previous_games)
            away_team_win_pct_at_away = len([g for g in away_team_previous_games if
                                             int(g.away_team.scores.get(GamePeriod.TOTAL)) > int(
                                                 g.home_team.scores.get(GamePeriod.TOTAL))]) / len(
                away_team_previous_games)
            return home_team_win_pct_at_home - away_team_win_pct_at_away
        except ZeroDivisionError:
            return 0.0

    def determine_experience_spread(self, game: Game) -> float:
        """
        Determines the spread between the home teams players Experience levels and the away teams players Experience
        levels

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        home_team_experience = 0.0
        away_team_experience = 0.0
        for p in game.home_team.players:
            home_team_experience += p.experience
        home_team_experience = home_team_experience / len(game.home_team.players)
        for p in game.away_team.players:
            away_team_experience += p.experience
        away_team_experience = away_team_experience / len(game.away_team.players)
        return float(home_team_experience - away_team_experience)

    def get_team_record_differential(self, game: Game) -> float:
        """
        Returns the teams record differential at this point of the season (before this game) as a float of games won
        against all games played. This method may be used to normalize the effects of the other features. This may be
        desirable in instances where the disparity in record between the teams is large so the effects of other in
        game features are lowered. Higher values returned by this function mean that this game is a more noteworth
        example

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
            Essentially, this function returns  1 / (home teams record - away teams record)
        """
        home_team_previous_games = [g for g in self.sorted_games if g.date < game.date and g.season == game.season and (
                g.home_team.name == game.home_team.name or g.away_team.name == game.home_team.name)]
        away_team_previous_games = [g for g in self.sorted_games if g.date < game.date and g.season == game.season and (
                g.home_team.name == game.away_team.name or g.away_team.name == game.away_team.name)]
        home_team_wins = [g for g in home_team_previous_games if (
                g.home_team.name == game.home_team.name and int(g.home_team.scores.get(GamePeriod.TOTAL)) > int(
            g.away_team.scores.get(GamePeriod.TOTAL))) or (g.away_team.name == game.home_team.name and int(
            g.away_team.scores.get(GamePeriod.TOTAL)) > int(g.home_team.scores.get(GamePeriod.TOTAL)))]
        away_team_wins = [g for g in away_team_previous_games if (
                g.home_team.name == game.away_team.name and int(g.home_team.scores.get(GamePeriod.TOTAL)) > int(
            g.away_team.scores.get(GamePeriod.TOTAL))) or (g.away_team.name == game.home_team.name and int(
            g.away_team.scores.get(GamePeriod.TOTAL)) > int(g.home_team.scores.get(GamePeriod.TOTAL)))]
        try:
            # If the difference between home record and away record is large in either direction, then we want a LOW
            # weight for this games features since the features may not have as much determination in the outcome
            home_team_record = len(home_team_wins) / len(home_team_previous_games)
            away_team_record = len(away_team_wins) / len(away_team_previous_games)
            return 1 / (home_team_record - away_team_record)
        except ZeroDivisionError:
            return 1.0

    def determine_rebound_differential(self, game: Game) -> float:
        """
        Determines the difference for rebounds between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.TRB) - game.away_team.team_stats.get(PlayerStatTypes.TRB)

    def determine_offensive_rebound_differential(self, game: Game) -> float:
        """
        Determines the difference for offensive rebounds between the two teams.

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.ORB) - game.away_team.team_stats.get(PlayerStatTypes.ORB)

    def determine_defensive_rebound_differential(self, game: Game) -> float:
        """
        Determines the difference for defensive rebounds between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.DRB) - game.away_team.team_stats.get(PlayerStatTypes.DRB)

    def determine_assist_spread(self, game: Game) -> float:
        """
        Determines the difference for assists between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.AST) - game.away_team.team_stats.get(PlayerStatTypes.AST)

    def determine_turnover_spread(self, game: Game) -> float:
        """
        Determines the difference for turnovers between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.TOV) - game.away_team.team_stats.get(PlayerStatTypes.TOV)

    def determine_field_goal_percent_spread(self, game: Game) -> float:
        """
        Determines the difference for field goal percentage between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.FGP) - game.away_team.team_stats.get(PlayerStatTypes.FGP)

    def determine_three_point_percent_spread(self, game: Game) -> float:
        """
        Determines the difference for three point percentage between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.THREEPP) - game.away_team.team_stats.get(
            PlayerStatTypes.THREEPP)

    def determine_free_throw_percent_spread(self, game: Game) -> float:
        """
        Determines the difference for free throw percentage between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.FTP) - game.away_team.team_stats.get(PlayerStatTypes.FTP)

    def determine_field_goals_attempted_spread(self, game: Game) -> float:
        """
        Determines the difference for field goals attempted between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.FGA) - game.away_team.team_stats.get(PlayerStatTypes.FGA)

    def determine_three_points_attempted_spread(self, game: Game) -> float:
        """
        Determines the difference for three points attempted between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.THREEPA) - game.away_team.team_stats.get(
            PlayerStatTypes.THREEPA)

    def determine_free_throws_attempted_spread(self, game: Game) -> float:
        """
        Determines the difference for free throws attempted between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.FTA) - game.away_team.team_stats.get(PlayerStatTypes.FTA)

    def determine_steals_spread(self, game: Game) -> float:
        """
        Determines the difference for steals between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.STL) - game.away_team.team_stats.get(PlayerStatTypes.STL)

    def determine_blocks_spread(self, game: Game) -> float:
        """
        Determines the difference for blocks between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.BLK) - game.away_team.team_stats.get(PlayerStatTypes.BLK)

    def determine_personal_foulds_spread(self, game: Game) -> float:
        """
        Determines the difference for personal fouls between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.PF) - game.away_team.team_stats.get(PlayerStatTypes.PF)

    def determine_true_shooting_percent_spread(self, game: Game) -> float:
        """
        Determines the difference for true shooting percentage between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.TSP) - game.away_team.team_stats.get(PlayerStatTypes.TSP)

    def determine_three_point_rate_spread(self, game: Game) -> float:
        """
        Determines the difference for three point rate between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.THREEPAR) - game.away_team.team_stats.get(
            PlayerStatTypes.THREEPAR)

    def determine_free_throw_rate_spread(self, game: Game) -> float:
        """
        Determines the difference for free throw rate between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.FTR) - game.away_team.team_stats.get(PlayerStatTypes.FTR)

    def determine_offensive_rating_spread(self, game: Game) -> float:
        """
        Determines the difference for offensive rating between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.ORTG) - game.away_team.team_stats.get(PlayerStatTypes.ORTG)

    def determine_defensive_rating_spread(self, game: Game) -> float:
        """
        Determines the difference for defensive rating between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.DRTF) - game.away_team.team_stats.get(PlayerStatTypes.DRTF)

    def determine_assist_to_turnover_spread(self, game: Game) -> float:
        """
        Determines the difference for assist-to-turnover-ratio between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.ASTTOV) - game.away_team.team_stats.get(
            PlayerStatTypes.ASTTOV)

    def determine_steal_to_turnover_spread(self, game: Game) -> float:
        """
        Determines the difference for steal-to-turnover-ratio between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.STLTOV) - game.away_team.team_stats.get(
            PlayerStatTypes.STLTOV)

    def determine_hob_spread_spread(self, game: Game) -> float:
        """
        Determines the difference for hands on ball between the two teams

        Parameters
        ----------
        game: Game

        Returns
        -------
        float
        """
        return game.home_team.team_stats.get(PlayerStatTypes.HOB) - game.away_team.team_stats.get(PlayerStatTypes.HOB)
