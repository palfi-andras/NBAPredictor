import pickle
from random import randint
from typing import Dict, List, Set

from game import Game
from player import Player

DEFAULT_SEASON = "2017-2018"


class League:

    def __init__(self, seasons_dict: Dict[str, List[Game]]):
        self.seasons_dict = seasons_dict
        self.all_players = self.generate_set_of_all_players()

    def save_league(self, path: str):
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def generate_set_of_all_players(self) -> Set[Player]:
        """
        :return: List of all players in the league at any time
        """
        all_players: Set[Player] = set()
        for season in self.seasons_dict.values():
            for game in season:
                for player in game.home_team.players:
                    all_players.add(player)
                for player in game.away_team.players:
                    all_players.add(player)
        return all_players

    def get_all_player_names(self) -> Set[str]:
        all_names: Set[str] = set()
        for player in self.all_players:
            all_names.add(player.name)
        return all_names

    def get_players_for_season(self, season: str) -> Set[Player]:
        assert season in self.seasons_dict.keys(), f"{season} not in the list of parsed seasons: " \
                                                   f"{self.seasons_dict.keys()}"
        players: Set[Player] = set()
        for game in self.seasons_dict[season]:
            for player in game.home_team.players:
                players.add(player)
            for player in game.away_team.players:
                players.add(player)
        return players

    def get_games_for_season(self, season: str) -> List[Game]:
        assert season in self.seasons_dict.keys(), f"{season} not in the list of parsed seasons: " \
                                                   f"{self.seasons_dict.keys()}"
        games: List[Game] = list()
        for game in self.seasons_dict[season]:
            games.append(game)
        return games

    def generate_player_averages_per_season(self, season: str) -> Set[Player]:
        assert season in self.seasons_dict.keys(), f"{season} not in the list of parsed seasons: " \
                                                   f"{self.seasons_dict.keys()}"
        players: Set[Player] = set()
        for game in self.seasons_dict[season]:
            for player in game.home_team.players:
                pass

    def get_random_game(self, season: str) -> Game:
        return self.seasons_dict[season][randint(0, len(self.seasons_dict[season]))]


def load_league(path: str) -> League:
    with open(path, 'rb') as pickle_file:
        league: League = pickle.load(pickle_file)
    return league
