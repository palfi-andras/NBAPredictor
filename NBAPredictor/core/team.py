from typing import Dict, List

from game_period import GamePeriod
from player import Player
from player_stat_types import PlayerStatTypes


class Team:

    def __init__(self, name: str, scores: Dict[str, List[Dict[GamePeriod, int]]],
            team_stats: Dict[str, List[Dict[PlayerStatTypes, float]]], players: Dict[str, List[Player]]):
        self.name = name
        self.scores = scores
        self.team_stats = team_stats
        self.players = players
