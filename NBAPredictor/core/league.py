import pickle
from typing import Dict, List

from game import Game
from player import Player


class League:

    def __init__(self, players: Dict[str, Player] = None, games: Dict[str, List[Game]] = None):
        self.games = games
        self.players = players

    def save_league(self, path: str):
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def players_exists(self, player: str) -> bool:
        return player in self.players


def load_league(path: str) -> League:
    with open(path, 'rb') as pickle_file:
        league: League = pickle.load(pickle_file)
    return league
