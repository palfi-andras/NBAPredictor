import re
from typing import List

import numpy as np

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
                     "StealToTurnoverSpread", "HOBSpread"]

DEFAULT_FEATURES = ["ReboundSpread", "OffensiveReboundSpread", "DefensiveReboundSpread", "AssistSpread",
                    "TurnoverSpread", "FieldGoalPercentSpread", "ThreePointPercentSpread", "FreeThrowPercentSpread",
                    "FieldGoalsAttemptedSpread", "ThreePointsAttemptedSpread", "FreeThrowsAttemptedSpread",
                    "BestPlayerSpread"]


def determine_best_player_from_team(team: Team) -> Player:
    best_val = float('-inf')
    best_player = None
    for player in team.players:
        fic = player.stats.get(PlayerStatTypes.FIC)
        if fic is not None:
            if fic > best_val:
                best_player = player
    return best_player


class ReadGames:

    def __init__(self, leauge: League, season: str, split: float, features: List[str] = DEFAULT_FEATURES):
        self.leauge = leauge
        assert season in self.leauge.seasons_dict, f"Cant find season {season}"
        assert all(features) in POSSIBLE_FEATURES
        self.features = features
        self.training_size = round(len(self.leauge.seasons_dict[season]) * split)
        self.sorted_games = sorted(self.leauge.seasons_dict[season].__iter__(),
                                   key=lambda x: re.sub(r"[A-Z]", "", x.code))
        self.training_features, self.training_labels, self.testing_features, self.testing_labels = \
            self.parse_whole_season()

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
            training_labels.append(self.get_winner(game))
            for feature in self.features:
                training_features[feature].append(self.map_feature_name_to_actual_value(feature, game))
        for x in range(self.training_size + 1, len(self.sorted_games)):
            game = self.sorted_games[x]
            testing_labels.append(self.get_winner(game))
            for feature in self.features:
                testing_features[feature].append(self.map_feature_name_to_actual_value(feature, game))
        for item in training_features:
            training_features[item] = np.array(training_features[item])
        for item in testing_features:
            testing_features[item] = np.array(testing_features[item])
        training_labels = np.array([label for label in training_labels])
        testing_labels = np.array([label for label in testing_labels])
        print(f"Number of games used for training: {len(training_labels)}")
        print(f"Number of games used for testing: {len(testing_labels)}")
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

    def get_team_record_differential(self, game: Game) -> float:
        """
        Returns the teams record differential at this point of the season (before this game) as a float of games won
        against all
        games played
        """
        home_team_previous_games = [g for g in self.sorted_games if g.date < game.date and (
                g.home_team == game.home_team or g.away_team == game.home_team)]
        away_team_previous_games = [g for g in self.sorted_games if g.date < game.date and (
                g.home_team == game.away_team or g.away_team == game.away_team)]
        home_team_wins = [g for g in home_team_previous_games if (
                g.home_team == game.home_team and g.home_team.scores.get(GamePeriod.TOTAL) > g.away_team.scores.get(
            GamePeriod.TOTAL)) or (g.away_team == game.home_team and g.away_team.scores.get(
            GamePeriod.TOTAL) > g.home_team.scores.get(GamePeriod.TOT))]
        away_team_wins = [g for g in away_team_previous_games if (
                g.home_team == game.home_team and g.home_team.scores.get(GamePeriod.TOTAL) > g.away_team.scores.get(
            GamePeriod.TOTAL)) or (g.away_team == game.home_team and g.away_team.scores.get(
            GamePeriod.TOTAL) > g.home_team.scores.get(GamePeriod.TOT))]
        home_team_record = len(home_team_wins) / len(home_team_previous_games)
        away_team_record = len(away_team_wins) / len(away_team_previous_games)
        return home_team_record - away_team_record

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
        return game.home_team.team_stats.get(PlayerStatTypes.DRTG) - game.away_team.team_stats.get(PlayerStatTypes.DRTG)

    def determine_assist_to_turnover_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.ASTTOV) - game.away_team.team_stats.get(
            PlayerStatTypes.ASTTOV)

    def determine_steal_to_turnover_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.STLTOV) - game.away_team.team_stats.get(
            PlayerStatTypes.STLTOV)

    def determine_hob_spread_spread(self, game: Game) -> float:
        return game.home_team.team_stats.get(PlayerStatTypes.HOB) - game.away_team.team_stats.get(PlayerStatTypes.HOB)
