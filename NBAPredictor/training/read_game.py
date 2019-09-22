import re
from typing import List

import numpy as np

from game import Game
from game_period import GamePeriod
from league import League, DEFAULT_SEASON
from player import Player
from player_stat_types import PlayerStatTypes
from team import Team


def determine_best_player_from_team(team: Team) -> Player:
    best_val = float('-inf')
    best_player = None
    for player in team.players:
        fic = player.stats.get(PlayerStatTypes.FIC)
        if fic is not None:
            if fic > best_val:
                best_player = player
    return best_player


def build_input_labels_array() -> List[str]:
    return ["ReboundSpread", "OffensiveReboundSpread", "DefensiveReboundSpread", "AssistSpread", "TurnoverSpread",
            "FieldGoalPercentSpread", "ThreePointPercentSpread", "FreeThrowPercentSpread", "FieldGoalsAttemptedSpread",
            "ThreePointsAttemptedSpread", "FreeThrowsAttemptedSpread", "BestPlayerSpread"]


class ReadGames:

    def __init__(self, leauge: League, season: str = DEFAULT_SEASON, split: float = 0.8):
        self.leauge = leauge
        assert season in self.leauge.seasons_dict, f"Cant find season {season}"
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
        for label in build_input_labels_array():
            training_features.setdefault(label, list())
            testing_features.setdefault(label, list())
        for x in range(0, self.training_size):
            game = self.sorted_games[x]
            training_labels.append(self.get_winner(game))
            training_features["ReboundSpread"].append(self.determine_rebound_differential(game))
            training_features["OffensiveReboundSpread"].append(self.determine_offensive_rebound_differential(game))
            training_features["DefensiveReboundSpread"].append(self.determine_defensive_rebound_differential(game))
            training_features["AssistSpread"].append(self.determine_assist_spread(game))
            training_features["TurnoverSpread"].append(self.determine_turnover_spread(game))
            training_features["FieldGoalPercentSpread"].append(self.determine_field_goal_percent_spread(game))
            training_features["ThreePointPercentSpread"].append(self.determine_three_point_percent_spread(game))
            training_features["FreeThrowPercentSpread"].append(self.determine_free_throw_percent_spread(game))
            training_features["FieldGoalsAttemptedSpread"].append(self.determine_field_goals_attempted_spread(game))
            training_features["ThreePointsAttemptedSpread"].append(self.determine_three_points_attempted_spread(game))
            training_features["FreeThrowsAttemptedSpread"].append(self.determine_free_throws_attempted_spread(game))
            training_features["BestPlayerSpread"].append(self.determine_best_player_spread(game))
        for x in range(self.training_size + 1, len(self.sorted_games)):
            game = self.sorted_games[x]
            testing_labels.append(self.get_winner(game))
            testing_features["ReboundSpread"].append(self.determine_rebound_differential(game))
            testing_features["OffensiveReboundSpread"].append(self.determine_offensive_rebound_differential(game))
            testing_features["DefensiveReboundSpread"].append(self.determine_defensive_rebound_differential(game))
            testing_features["AssistSpread"].append(self.determine_assist_spread(game))
            testing_features["TurnoverSpread"].append(self.determine_turnover_spread(game))
            testing_features["FieldGoalPercentSpread"].append(self.determine_field_goal_percent_spread(game))
            testing_features["ThreePointPercentSpread"].append(self.determine_three_point_percent_spread(game))
            testing_features["FreeThrowPercentSpread"].append(self.determine_free_throw_percent_spread(game))
            testing_features["FieldGoalsAttemptedSpread"].append(self.determine_field_goals_attempted_spread(game))
            testing_features["ThreePointsAttemptedSpread"].append(self.determine_three_points_attempted_spread(game))
            testing_features["FreeThrowsAttemptedSpread"].append(self.determine_free_throws_attempted_spread(game))
            testing_features["BestPlayerSpread"].append(self.determine_best_player_spread(game))
        for item in training_features:
            training_features[item] = np.array(training_features[item])
        for item in testing_features:
            testing_features[item] = np.array(testing_features[item])
        training_labels = np.array([label for label in training_labels])
        testing_labels = np.array([label for label in testing_labels])
        print(f"Number of games used for training: {len(training_labels)}")
        print(f"Number of games used for testing: {len(testing_labels)}")
        return training_features, training_labels, testing_features, testing_labels

    def get_winner(self, game: Game):
        home_team_score = game.home_team.scores.get(GamePeriod.TOTAL)
        away_team_score = game.away_team.scores.get(GamePeriod.TOTAL)
        return 'H' if home_team_score > away_team_score else 'A'

    def determine_best_player_spread(self, game: Game) -> float:
        home_team_best_player = determine_best_player_from_team(game.home_team)
        away_team_best_player = determine_best_player_from_team(game.away_team)
        return home_team_best_player.stats.get(PlayerStatTypes.FIC) - away_team_best_player.stats.get(
            PlayerStatTypes.FIC)

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
