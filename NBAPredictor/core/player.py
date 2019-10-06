from typing import Dict

from player_stat_types import PlayerStatTypes
from positions import Position


class Player:
    """
    A base class to represent an NBA player and his attributes in one NBA game. Please note, multiple Player objects
    may exist in NBAPredictor for the same player, as these represent Player individual performances

    Attributes
    ----------
    name: str
        The name of the Player
    experience: int
        The amount of years this player has been in the league
    position: Position
        The Position this Player plays
    height: float
        The height of this Player in inches
    weight: str
        The weight of this player in pounds
    stats: dict
        A dictionary of stats and values that this player achieved for a singular NBA game.
    """

    def __init__(self, name: str, experience: int, position: Position, height: float, weight: str,
            stats: Dict[PlayerStatTypes, float]):
        """
        Parameters
        ----------
        name: str
            The name of the Player
        experience: int
            The amount of years this player has been in the league
        position: Position
            The Position this Player plays
        height: float
            The height of this Player in inches
        weight: str
            The weight of this player in pounds
        stats: dict
            A dictionary of stats and values that this player achieved for a singular NBA game.
        """
        self.name = name
        self.experience = experience
        self.position = position
        self.height = height
        self.weight = weight
        self.stats = stats
