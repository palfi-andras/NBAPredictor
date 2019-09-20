from game import Game
import numpy as np
from team import Team
from typing import List, Dict
from player import Player
from player_stat_types import PlayerStatTypes


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
            "FieldGoal%Spread", "ThreePoint%Spread", "FreeThrow%Spread", "FieldGoalsAttemptedSpread",
            "ThreePointsAttemptedSpread", "FreeThrowsAttemptedSpread", "BestPlayerSpread"]


class ReadGame:

    def __init__(self, game: Game):
        self.game = game
        self.labels = build_input_labels_array()
        self.inputs = self.build_input_array()

    def build_input_array(self) -> np.array.__class__:
        """
        Returns a Numpy array that can be used as input into a the neural network for this application. It returns
        a nump array with indexes defined exactly as it appears in the build_input_labels_array function above
        """
        return np.array([self.determine_rebound_differential(),
                         self.determine_offensive_rebound_differential(),
                         self.determine_defensive_rebound_differential(),
                         self.determine_assist_spread(),
                         self.determine_turnover_spread(),
                         self.determine_field_goal_percent_spread(),
                         self.determine_three_point_percent_spread(),
                         self.determine_free_throw_percent_spread(),
                         self.determine_field_goals_attempted_spread(),
                         self.determine_three_points_attempted_spread(),
                         self.determine_free_throws_attempted_spread(),
                         self.determine_best_player_spread()])

    def determine_best_player_spread(self) -> float:
        home_team_best_player = determine_best_player_from_team(self.game.home_team)
        away_team_best_player = determine_best_player_from_team(self.game.away_team)
        return home_team_best_player.stats.get(PlayerStatTypes.FIC) - away_team_best_player.stats.get(
            PlayerStatTypes.FIC)

    def determine_rebound_differential(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.TRB) - self.game.away_team.team_stats.get(
            PlayerStatTypes.TRB)

    def determine_offensive_rebound_differential(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.ORB) - self.game.away_team.team_stats.get(
            PlayerStatTypes.ORB)

    def determine_defensive_rebound_differential(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.DRB) - self.game.away_team.team_stats.get(
            PlayerStatTypes.DRB)

    def determine_assist_spread(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.AST) - self.game.away_team.team_stats.get(
            PlayerStatTypes.AST)

    def determine_turnover_spread(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.TOV) - self.game.away_team.team_stats.get(
            PlayerStatTypes.TOV)

    def determine_field_goal_percent_spread(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.FGP) - self.game.away_team.team_stats.get(
            PlayerStatTypes.FGP)

    def determine_three_point_percent_spread(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.THREEPP) - self.game.away_team.team_stats.get(
            PlayerStatTypes.THREEPP)

    def determine_free_throw_percent_spread(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.FTP) - self.game.away_team.team_stats.get(
            PlayerStatTypes.FTP)

    def determine_field_goals_attempted_spread(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.FGA) - self.game.away_team.team_stats.get(
            PlayerStatTypes.FGA)

    def determine_three_points_attempted_spread(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.THREEPA) - self.game.away_team.team_stats.get(
            PlayerStatTypes.THREEPA)

    def determine_free_throws_attempted_spread(self) -> float:
        return self.game.home_team.team_stats.get(PlayerStatTypes.FTA) - self.game.away_team.team_stats.get(
            PlayerStatTypes.FTA)

