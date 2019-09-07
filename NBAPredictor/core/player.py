from typing import Dict

from player_stat_types import PlayerStatTypes
from positions import Position


class Player:

    def __init__(self, name: str, experience: int, position: Position, height: float, weight: str,
            stats: Dict[PlayerStatTypes, float]):
        self.name = name
        self.experience = experience
        self.position = position
        self.height = height
        self.weight = weight
        self.stats = stats
