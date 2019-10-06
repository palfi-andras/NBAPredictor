import pickle
from collections import OrderedDict
from random import randint
from typing import Dict, List, Set

from game import Game
from player import Player

DEFAULT_SEASON = "2017-2018"


class League:
    """
    A class used to represent an entire NBA season. A season in this context means the seasons played between the years
    2000 to the present day.

    Attributes
    ----------
    seasons_dict: dict
        A dictionary where the keys are a particular NBA season followed by a list of Game objects for all valid
        games in that season

    all_players: set
        Set of all players in this league

    Methods
    -------
    save_league (path: str)
        Writes this object out as a pickle dump so that the JSONs for NBA data do not need to be parsed again and this
        object cant be reused in future executions of this program.

    generate_set_of_all_players
        Generates a set of all players that have ever played in the league
    """

    def __init__(self, seasons_dict: Dict[str, List[Game]]):
        """
        Parameters
        ----------
        seasons_dict: dict
            A dictionary where the keys are a particular NBA season followed by a list of Game objects for all valid
            games in that season
        """
        self.seasons_dict = seasons_dict
        self.all_players = self.generate_set_of_all_players()

    def save_league(self, path: str) -> None:
        """
        Writes this object out as a pickle dump to path so that the JSONs for NBA data do not need to be parsed
            again and this object cant be reused in future executions of this program.

        Parameters
        ----------
        path: str
            The path to write the pickle object to

        Returns
        -------
        None
        """
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def generate_set_of_all_players(self) -> Set[Player]:
        """
        Generates a set of all players that have ever played in the league

        Returns
        -------
        all_players: set
            The set of all Player objects for each player in the league
        """
        all_players: Set[Player] = set()
        for season in self.seasons_dict.values():
            for game in season:
                for player in game.home_team.players:
                    all_players.add(player)
                for player in game.away_team.players:
                    all_players.add(player)
        return all_players


def load_league(path: str) -> League:
    """
    An external function that can be used by handlers of this class to load a previously saved League object as a
    Pickle dump. This will preform a pickle load on that data.

    Parameters
    ----------
    path: str
        The path to the pickle object to load

    Returns
    -------
    league: League
        The league object of the converted Pickle data
    """
    with open(path, 'rb') as pickle_file:
        league: League = pickle.load(pickle_file)
    return league
