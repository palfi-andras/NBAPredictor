import re
from typing import List
import logging
import time

import numpy as np
from statistics import stdev

from game import Game
from game_period import GamePeriod
from league import League
from player import Player
from player_stat_types import PlayerStatTypes
from team import Team

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
    best_val = float('-inf')
    best_player = None
    for player in team.players:
        fic = player.stats.get(PlayerStatTypes.FIC)
        if fic is not None:
            if fic > best_val:
                best_player = player
    return best_player


def center_values(values: List[float]) -> List[float]:
    try:
        avg = sum(values) / len(values)
        return [(x - avg) / stdev(values) for x in values]
    except ZeroDivisionError:
        return values


def center_numpy_features(values: np.array, normalize=True) -> np.array:
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
    start_year = int(seasons[2:4])
    end_year = int(seasons[-2:])
    season_list = list()
    if (end_year - start_year) > 1:
        num_years = end_year - start_year
        for x in range(start_year, start_year + num_years):
            season_list.append(f"20{x}-20{x + 1}")
    else:
        season_list.append(seasons)
    return season_list


class ReadGames:

    def __init__(self, leauge: League, season: str, split: float, logger: logging,
            features: List[str] = DEFAULT_FEATURES, svm_compat=False, normalize_weights=False):
        start_time = time.time()
        self.leauge = leauge
        self.logger = logger
        self.normalize_weights = normalize_weights
        seasons = [s for i, s in enumerate(check_for_multiple_seasons(season)) if s in self.leauge.seasons_dict]
        self.logger.info(f"Using the following NBA seasons: {seasons}")
        assert all(e in POSSIBLE_FEATURES for e in features)
        self.features = features
        if len(seasons) == 1:
            self.training_size = round(len(self.leauge.seasons_dict[season]) * split)
            self.sorted_games = sorted(self.leauge.seasons_dict[season].__iter__(),
                                       key=lambda x: re.sub(r"[A-Z]", "", x.code))
        else:
            self.training_size = round(sum(len(self.leauge.seasons_dict[s]) for s in seasons) * split)
            self.sorted_games = list()
            for s in seasons:
                self.sorted_games.extend(
                    sorted(self.leauge.seasons_dict[s].__iter__(), key=lambda x: re.sub(r"[A-Z]", "", x.code)))
        if not svm_compat:
            self.training_features, self.training_labels, self.testing_features, self.testing_labels = \
                self.parse_whole_season()
        else:
            self.training_features, self.training_labels, self.testing_features, self.testing_labels = \
                self.parse_whole_season_svm_format()
        self.logger.info(f"Prepared all data sets in {time.time() - start_time} seconds")

    def parse_whole_season(self):
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
            return 0.0

    def get_winner(self, game: Game):
        home_team_score = game.home_team.scores.get(GamePeriod.TOTAL)
        away_team_score = game.away_team.scores.get(GamePeriod.TOTAL)
        return 'H' if home_team_score > away_team_score else 'A'

    def determine_best_player_spread(self, game: Game) -> float:
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
      Calculates the difference ebtween each players experience levels
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
        against all games played

        This method may be used to normalize the effects of the other features. This may be desirable in instances where
        the disparity in record between the teams is large so the effects of other in game features are lowered.
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
        return game.home_team.team_stats.get(PlayerStatTypes.TRB) - game.away_team.team_stats.get(PlayerStatTypes.TRB)

    def determine_offensive_rebound_differential(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.ORB) - game.away_team.team_stats.get(PlayerStatTypes.ORB)

    def determine_defensive_rebound_differential(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.DRB) - game.away_team.team_stats.get(PlayerStatTypes.DRB)

    def determine_assist_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.AST) - game.away_team.team_stats.get(PlayerStatTypes.AST)

    def determine_turnover_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.TOV) - game.away_team.team_stats.get(PlayerStatTypes.TOV)

    def determine_field_goal_percent_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.FGP) - game.away_team.team_stats.get(PlayerStatTypes.FGP)

    def determine_three_point_percent_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.THREEPP) - game.away_team.team_stats.get(
            PlayerStatTypes.THREEPP)

    def determine_free_throw_percent_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.FTP) - game.away_team.team_stats.get(PlayerStatTypes.FTP)

    def determine_field_goals_attempted_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.FGA) - game.away_team.team_stats.get(PlayerStatTypes.FGA)

    def determine_three_points_attempted_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.THREEPA) - game.away_team.team_stats.get(
            PlayerStatTypes.THREEPA)

    def determine_free_throws_attempted_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.FTA) - game.away_team.team_stats.get(PlayerStatTypes.FTA)

    def determine_steals_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.STL) - game.away_team.team_stats.get(PlayerStatTypes.STL)

    def determine_blocks_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.BLK) - game.away_team.team_stats.get(PlayerStatTypes.BLK)

    def determine_personal_foulds_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.PF) - game.away_team.team_stats.get(PlayerStatTypes.PF)

    def determine_true_shooting_percent_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.TSP) - game.away_team.team_stats.get(PlayerStatTypes.TSP)

    def determine_three_point_rate_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.THREEPAR) - game.away_team.team_stats.get(
            PlayerStatTypes.THREEPAR)

    def determine_free_throw_rate_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.FTR) - game.away_team.team_stats.get(PlayerStatTypes.FTR)

    def determine_offensive_rating_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.ORTG) - game.away_team.team_stats.get(PlayerStatTypes.ORTG)

    def determine_defensive_rating_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.DRTF) - game.away_team.team_stats.get(PlayerStatTypes.DRTF)

    def determine_assist_to_turnover_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.ASTTOV) - game.away_team.team_stats.get(
            PlayerStatTypes.ASTTOV)

    def determine_steal_to_turnover_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.STLTOV) - game.away_team.team_stats.get(
            PlayerStatTypes.STLTOV)

    def determine_hob_spread_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.HOB) - game.away_team.team_stats.get(PlayerStatTypes.HOB)
