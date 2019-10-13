import hashlib
import logging
import os
import pickle
import random
from enum import Enum
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd

from league import League
from read_game import ReadGames


class PredictNextSeason(ReadGames):
    """
    A class that extends ReadGames(therefore, basically acting as a container for NumPy Games) but the difference is
    that this class will test upon the next NBA Season (2019-2020). It will simulate the entire season and then
    simulate the playoffs as well.
    """

    def __init__(self, next_season_csv: str, leauge: League, season: str, split: float, logger: logging, cache_dir: str,
            features: List[str], normalize_weights=False, cache=False):
        super().__init__(leauge=leauge, season=season, split=split, logger=logger, cache_dir=cache_dir,
                         features=features, svm_compat=False, normalize_weights=normalize_weights, cache=cache,
                         initialize=False)
        self.next_season_csv = next_season_csv
        self.next_season, self.conferences = self.parse_next_season_csv_file()
        self.averages = None
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

    def parse_next_season_csv_file(self) -> Tuple[List[Dict], Dict[str, List]]:
        assert os.path.isfile(self.next_season_csv), f"Cant find file {self.next_season_csv}"
        schedule = pd.read_excel(self.next_season_csv, sheet_name=0)
        teams = pd.read_excel(self.next_season_csv, sheet_name=1)
        return [{"home": row[6], "away": row[9]} for index, row in schedule.iterrows() if index > 3], {
            "East": [row[0] for index, row in teams.iterrows() if index > 1 and row[4] == "East"],
            "West": [row[0] for index, row in teams.iterrows() if index > 1 and row[4] == "West"]}

    def load_instance(self, dir: str) -> Union[Tuple, None]:
        assert self.seasons and self.features and self.training_size and self.sorted_games
        name = f"next-season-instance-{self.training_size}-{len(self.sorted_games)}-" \
               f"{str(hashlib.sha1(str(self.seasons).encode('utf-8')).hexdigest())}-" \
               f"{str(hashlib.sha1(str(self.features).encode('utf-8')).hexdigest())}"
        path = os.path.join(dir, name)
        if os.path.isfile(path):
            with open(path, 'rb') as pickle_file:
                instance: PredictNextSeason = pickle.load(pickle_file)
                return instance.training_features, instance.training_labels, instance.testing_features, \
                       instance.testing_labels
        return None

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
        name = f"next-season-instance-{self.training_size}-{len(self.sorted_games)}-" \
               f"{str(hashlib.sha1(str(self.seasons).encode('utf-8')).hexdigest())}-" \
               f"{str(hashlib.sha1(str(self.features).encode('utf-8')).hexdigest())}"
        path = os.path.join(dir, name)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

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
        averages = dict()
        training_features = dict()
        training_labels = np.array([])
        testing_features = dict()
        testing_labels = list()

        for game in self.sorted_games:
            self.logger.info(f"Extracting requested features for game {game.code}")
            training_labels = np.append(training_labels, [self.get_winner(game)])
            averages.setdefault(game.home_team.name, dict())
            averages.setdefault(game.away_team.name, dict())
            for feature in self.features:
                training_features.setdefault(feature, np.array([]))
                feature_value = self.map_feature_name_to_actual_value(feature, game)
                training_features[feature] = np.append(training_features[feature], [feature_value])
                averages[game.home_team.name].setdefault(feature, 0.0)
                averages[game.away_team.name].setdefault(feature, 0.0)
                averages[game.home_team.name][feature] += feature_value
                averages[game.away_team.name][feature] += feature_value
        for game in self.next_season:
            testing_labels = np.append(testing_labels, [random.choice(['H', 'A', 'H'])])
            home = game["home"]
            away = game["away"]
            self.logger.info(f"Building dataset for future 2019-2020 NBA game between {home} and {away}")
            try:
                home_team_dict = averages[home]
            except KeyError:
                self.logger.error(f"Cant find team key name: {home}")
                exit(1)
            try:
                away_team_dict = averages[away]
            except KeyError:
                self.logger.error(f"Cant find team key name: {away}")
                exit(1)
            for feature in self.features:
                testing_features.setdefault(feature, np.array([]))
                home_feature_value = home_team_dict[feature] / len(self.sorted_games)
                away_feature_value = away_team_dict[feature] / len(self.sorted_games)
                testing_features[feature] = np.append(testing_features[feature],
                                                      [home_feature_value - away_feature_value])
        self.averages = averages
        return training_features, training_labels, testing_features, testing_labels

    def analyze_end_of_season_predictions(self, predictions: list):
        win_tracker = dict()
        playoff_teams = dict()
        playoff_teams.setdefault("West", list())
        playoff_teams.setdefault("East", list())
        for i, game in enumerate(self.next_season):
            win_tracker.setdefault(game['home'], 0)
            win_tracker.setdefault(game['away'], 0)
            probabilities = predictions[i]['probabilities']
            winner = game['home'] if probabilities[0] > probabilities[1] else game['away']
            win_tracker[winner] += 1
            self.logger.info(
                f"For future NBA game between {game['home']} and {game['away']}, predicting the winner to be "
                f"{winner}")
        for team in win_tracker:
            self.logger.info(f"Predicted {team} record for 2019-2020 NBA Season to be: {win_tracker[team]}-"
                             f"{81 - win_tracker[team]}")
        east_sorted = [team for team in sorted(win_tracker, key=lambda x: win_tracker[x]) if
                       team in self.conferences["East"]]
        west_sorted = [team for team in sorted(win_tracker, key=lambda x: win_tracker[x]) if
                       team in self.conferences["West"]]
        playoff_teams["East"] = [east_sorted[x - 1] for x in range(1, 9)]
        playoff_teams["West"] = [west_sorted[x - 1] for x in range(1, 9)]
        return Playoffs(playoff_teams, logger=self.logger, average_performance=self.averages, features=self.features,
                        sorted_games=self.sorted_games)


class Round(Enum):
    FIRST_ROUND = 1,
    QUARTERFINALS = 2,
    SEMIFINALS = 3,
    FINALS = 4


class Playoffs:

    def __init__(self, playoff_teams: Dict[str, List[str]], average_performance: Dict[str, Dict[str, float]],
            sorted_games: List, features: List[str], logger):
        self.logger = logger
        self.logger.info("2019-2020 NBA PLAYOFFS\n")
        self.logger.info("EASTERN CONFERENCE: ")
        self.seeding = {}
        for conf in playoff_teams:
            for i, team in enumerate(playoff_teams[conf]):
                self.seeding[team] = i + 1
        self.logger.info("1. {} vs 8. {}".format(playoff_teams["East"][0], playoff_teams["East"][-1]))
        self.logger.info("2. {} vs 7. {}".format(playoff_teams["East"][1], playoff_teams["East"][-2]))
        self.logger.info("3. {} vs 6. {}".format(playoff_teams["East"][2], playoff_teams["East"][-3]))
        self.logger.info("4. {} vs 5. {}".format(playoff_teams["East"][3], playoff_teams["East"][-4]))
        self.logger.info("WESTERN CONFERENCE: ")
        self.logger.info("1. {} vs 8. {}".format(playoff_teams["West"][0], playoff_teams["West"][-1]))
        self.logger.info("2. {} vs 7. {}".format(playoff_teams["West"][1], playoff_teams["West"][-2]))
        self.logger.info("3. {} vs 6. {}".format(playoff_teams["West"][2], playoff_teams["West"][-3]))
        self.logger.info("4. {} vs 5. {}".format(playoff_teams["West"][3], playoff_teams["West"][-4]))
        self.playoff_tree = {Round.FIRST_ROUND: {"East": [{playoff_teams["East"][0]: 0, playoff_teams["East"][-1]: 0},
                                                          {playoff_teams["East"][1]: 0, playoff_teams["East"][-2]: 0},
                                                          {playoff_teams["East"][2]: 0, playoff_teams["East"][-3]: 0},
                                                          {playoff_teams["East"][3]: 0, playoff_teams["East"][-4]: 0}],
                                                 "West": [{playoff_teams["West"][0]: 0, playoff_teams["West"][-1]: 0},
                                                          {playoff_teams["West"][1]: 0, playoff_teams["West"][-2]: 0},
                                                          {playoff_teams["West"][2]: 0, playoff_teams["West"][-3]: 0},
                                                          {playoff_teams["West"][3]: 0, playoff_teams["West"][-4]: 0}]},
                             Round.QUARTERFINALS: {"East": [], "West": []}, Round.SEMIFINALS: {"East": [], "West": []},
                             Round.FINALS: {}}
        self.sorted_games = sorted_games
        self.features = features
        self.averages = average_performance
        self.current_games = []
        self.current_round = Round.FIRST_ROUND

    def generate_test_data_for_playoffs(self):
        # Describes where games are played for each game in a 7-game series. 0 stands for home game, 1 for away
        PLAYOFF_FORMAT = [0, 0, 1, 1, 0, 1, 0]
        games = list()
        testing_features = dict()
        testing_labels = np.array([])
        for round, bracket in self.playoff_tree.items():
            round_finished = all(all(4 in g.values() for g in matchups) for matchups in
                                 bracket.values()) if round != Round.FINALS else 4 in self.playoff_tree[
                Round.FINALS].values()
            # Build the matchups for next round if this rounbd is finished
            if round_finished:
                if round != self.current_round:
                    continue
                if round == Round.FIRST_ROUND:
                    for conference, matchups in bracket.items():
                        winner_1 = sorted(matchups[0], key=lambda x: matchups[0][x], reverse=True)[0]
                        winner_2 = sorted(matchups[1], key=lambda x: matchups[1][x], reverse=True)[0]
                        winner_3 = sorted(matchups[2], key=lambda x: matchups[2][x], reverse=True)[0]
                        winner_4 = sorted(matchups[3], key=lambda x: matchups[3][x], reverse=True)[0]
                        self.playoff_tree[Round.QUARTERFINALS][conference].append({winner_1: 0, winner_4: 0})
                        self.playoff_tree[Round.QUARTERFINALS][conference].append({winner_2: 0, winner_3: 0})
                        self.current_round = Round.QUARTERFINALS
                    continue
                if round == Round.QUARTERFINALS and self.current_round == round:
                    for conference, matchups in bracket.items():
                        winner_1 = sorted(matchups[0], key=lambda x: matchups[0][x], reverse=True)[0]
                        winner_2 = sorted(matchups[1], key=lambda x: matchups[1][x], reverse=True)[0]
                        self.playoff_tree[Round.SEMIFINALS][conference].append({winner_1: 0, winner_2: 0})
                        self.current_round = Round.SEMIFINALS
                    continue
                if round == Round.SEMIFINALS and self.current_round == round:
                    winner_1 = None
                    winner_2 = None
                    for conference, matchups in bracket.items():
                        winner_1 = sorted(matchups[0], key=lambda x: matchups[0][x], reverse=True)[
                            0] if conference == "East" else winner_1
                        winner_2 = sorted(matchups[0], key=lambda x: matchups[0][x], reverse=True)[
                            0] if conference == "West" else winner_2
                    self.playoff_tree[Round.FINALS][winner_1] = 0
                    self.playoff_tree[Round.FINALS][winner_2] = 0
                    self.current_round = Round.FINALS
                    continue
                if round == Round.FINALS:
                    break
            # Build the games for the current matchups (only adding games if a matchup between two teams hasnt reach
            # 4 wins for either side yet)
            if round != Round.FINALS:
                for matchups in bracket.values():
                    for g in matchups:
                        if 4 in g.values():
                            continue
                        current_game = sum(g.values())
                        home_team = list(g.keys())[PLAYOFF_FORMAT[current_game]]
                        away_team = list(g.keys())[abs(PLAYOFF_FORMAT[current_game] - 1)]
                        games.append({"home": home_team, "away": away_team})
                break
            else:
                current_game = sum(bracket.values())
                home_team = list(bracket.keys())[PLAYOFF_FORMAT[current_game]]
                away_team = list(bracket.keys())[abs(PLAYOFF_FORMAT[current_game] - 1)]
                games.append({"home": home_team, "away": away_team})
        # Build the NumPy data for this round
        self.current_games = games
        for game in games:
            testing_labels = np.append(testing_labels, [random.choice(['H', 'A', 'H'])])
            home = game["home"]
            away = game["away"]
            self.logger.info(f"Building dataset for future 2019-2020 NBA game between {home} and {away}")
            try:
                home_team_dict = self.averages[home]
            except KeyError:
                self.logger.error(f"Cant find team key name: {home}")
                exit(1)
            try:
                away_team_dict = self.averages[away]
            except KeyError:
                self.logger.error(f"Cant find team key name: {away}")
                exit(1)
            for feature in self.features:
                testing_features.setdefault(feature, np.array([]))
                home_feature_value = home_team_dict[feature] / len(self.sorted_games)
                away_feature_value = away_team_dict[feature] / len(self.sorted_games)
                testing_features[feature] = np.append(testing_features[feature],
                                                      [home_feature_value - away_feature_value])
        if len(testing_labels) > 0:
            assert all(len(a) == len(testing_labels) for a in testing_features.values())
            return testing_features, testing_labels
        return False

    def record_playoff_results(self, predictions: List):
        assert len(self.current_games) == len(predictions)
        for i, game in enumerate(self.current_games):
            probabilities = predictions[i]['probabilities']
            winner = game['home'] if probabilities[0] > probabilities[1] else game['away']
            h_team = game['home']
            a_team = game['away']
            self.logger.info(f"Predicted winner of playoff game between {h_team} and {a_team} to be {winner}")
            for round, bracket in self.playoff_tree.items():
                if round != Round.FINALS:
                    for conf, matchups in bracket.items():
                        for x, g in enumerate(matchups):
                            if h_team in g.keys() and a_team in g.keys() and 4 not in g.values():
                                self.playoff_tree[round][conf][x][winner] += 1
                else:
                    if h_team in bracket.keys() and a_team in bracket.keys() and 4 not in bracket.values():
                        self.playoff_tree[round][winner] += 1

    def log_playoff_results(self):
        self.logger.info(f"###################################################################")
        self.logger.info(f"###################################################################")
        self.logger.info(f"************************* PLAYOFF RESULTS *************************")
        self.logger.info(f"###################################################################")
        self.logger.info(f"###################################################################\n\n")
        for round, bracket in self.playoff_tree.items():
            self.logger.info(f"                      {round.name.upper()}                                      ")
            if round != Round.FINALS:
                for conf, games in bracket.items():
                    self.logger.info(f"                      {conf.upper()}                                      ")
                    for game in games:
                        t1 = list(game.keys())[0]
                        t1_score = list(game.values())[0]
                        t1_rank = self.seeding[t1]
                        t2 = list(game.keys())[1]
                        t2_score = list(game.values())[1]
                        t2_rank = self.seeding[t2]
                        self.logger.info(f"\t\t\t{t1_rank}. {t1} vs {t2_rank}. {t2} : OUTCOME: {t1_score}-"
                                         f"{t2_score}\t\t\t")
            else:
                self.logger.info("\t\t\t2019-2020 NBA Season Predicted Finals")
                t1 = list(bracket.keys())[0]
                t1_score = list(bracket.values())[0]
                t1_rank = self.seeding[t1]
                t2 = list(bracket.keys())[1]
                t2_score = list(bracket.values())[1]
                t2_rank = self.seeding[t2]
                self.logger.info(f"\t\t\t{t1_rank}. {t1} vs {t2_rank}. {t2} : OUTCOME: {t1_score}-"
                                 f"{t2_score}\t\t\t")
                self.logger.info(f"\n\n2019-2020 NBA Champion: {t1 if t1_score > t2_score else t2}")
