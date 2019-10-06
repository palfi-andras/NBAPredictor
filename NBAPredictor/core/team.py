from typing import Dict, List

from game_period import GamePeriod
from player import Player
from player_stat_types import PlayerStatTypes


class Team:
    """
    A base class to represent a Team that played in a NBA game.

    Attributes
    ----------
    name: str
        The name of the team
    scores: dict
        A dictionary of GamePeriods to ints to signify what the score of the team was during each period of the game.
    team_stats: dict
        A dictionary of PlayerStateTypes to float to represent stat attributes this team achieved throughout the match.
    players: list
        A list of Player objects that played on this team
    """

    def __init__(self, name: str, scores: Dict[GamePeriod, int], team_stats: Dict[PlayerStatTypes, float],
            players: List[Player]):
        """
        Parameters
        ----------
        name: str
            The name of the team
        scores: dict
            A dictionary of GamePeriods to ints to signify what the score of the team was during each period of the
            game.
        team_stats: dict
            A dictionary of PlayerStateTypes to float to represent stat attributes this team achieved throughout the
            match.
        players: list
            A list of Player objects that played on this team
        """
        self.name = name
        self.scores = scores
        self.team_stats = team_stats
        self.players = players
