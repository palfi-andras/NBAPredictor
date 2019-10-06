import json
import os
import re
from typing import List, AnyStr, Dict, Any

from game import Game
from game_period import GamePeriod, convert_to_game_period
from league import League
from player import Player
from player_stat_types import PlayerStatTypes, convert_to_player_stat_type
from positions import Position, convert_to_position
from team import Team


class NBAJsonParser:
    """
    A class to parse NBA JSON files scraped from basketball-reference.com. This class is tasked with going through
    each JSON file (where each JSON file is data from any NBA game) and convert the data represented as text into the
    various objects represented in the core package such as League, Player, Game, etc so that they may be used by
    other parts of NBAPredictor.

    Attributes
    ----------
    nba_json_dir: str
        The path to the directory containing the NBA JSON data. This class expects that this directory already contains
        the JSON data that was created using the basketball_reference utility in NBAPRedictor/basketball_reference.
        Please see the help of that module to create the data files if not already there.

    games: list
        A list of all the Game objects that have been converted from JSON files

    Methods
    -------
    get_seasons
        Get the set of seasons that were successfully parsed by the basketball_reference parser

    get_all_game_files
        Get the list of paths to all the individual JSON files for NBA games

    parse_json_file (file: str)
        Loads a json file and returns a dictionary of the data

    convert_game_dict_to_core_objects (game_dict: dict)
        Loads a dictionary that represents one NBA game (i.e. one of the JSON files). Builds all of the required
        objects in the core package to represent that game

    generate_player_from_player_dict (player_dict: dict)
        Takes a dictionary of player data and creates a Player object out of it.

    generate_team_from_team_dict (team_dict: dict)
        Takes a dictionary of team data and creates a Team object out of it

    generate_league_object
        Calls other methods of this class in order to build a League object that holds all of the necessary Game,
        Player, and Team objects needed to accurately hold data for all games played in the NBA provided by the JSON
        files.

    """

    def __init__(self, nba_json_dir: str):
        """
        Parameters
        ----------
        nba_json_dir: str
            The path to the directory containing the NBA JSON data. This class expects that this directory already
            contains the JSON data that was created using the basketball_reference utility in
            NBAPRedictor/basketball_reference. Please see the help of that module to create the data files if not
            already there.
        """
        self.nba_json_dir: str = nba_json_dir
        assert os.path.isdir(self.nba_json_dir), f"{self.nba_json_dir} is not a valid directory"
        self.games = [self.convert_game_dict_to_core_objects(self.parse_json_file(game_file)) for game_file in
                      self.get_all_game_files()]

    def get_seasons(self) -> List[str]:
        """
        Get the set of seasons that were successfully parsed by the basketball_reference parser

        Returns
        -------
        seasons: list
            The list of strings that represent valid, parsed, NBA data
        """
        valid_season_name: AnyStr = r"20[0-1][-0-9]-20[0-1][-0-9]"
        seasons: List[str] = os.listdir(self.nba_json_dir)
        for season in seasons:
            if not re.match(valid_season_name, season):
                seasons.remove(season)
        return seasons

    def get_all_game_files(self) -> List[str]:
        """
        Get the list of paths to all the individual JSON files for NBA games

        Returns
        -------
        all_game_files: list
            Path names to individual NBA game data stored in JSON files
        """
        all_game_files: List[str] = list()
        for season in self.get_seasons():
            for game in os.listdir(os.path.join(self.nba_json_dir, season)):
                all_game_files.append(os.path.join(self.nba_json_dir, season, game))
        return all_game_files

    def parse_json_file(self, file: str) -> Dict[Any, Any]:
        """
        Loads a json file and returns a dictionary of the data

        Parameters
        ----------
        file: str
            Valid path to the JSON file

        Returns
        -------
        json_file: dict
            Dictionary of the data in the JSON file
        """
        assert os.path.isfile(file)
        with open(file, 'r') as file:
            json_file: Dict[Any, Any] = json.load(file)
        return json_file

    def convert_game_dict_to_core_objects(self, game_dict: Dict[Any, Any]) -> Game:
        """
        Loads a dictionary that represents one NBA game (i.e. one of the JSON files). Builds all of the required
        objects in the core package to represent that game

        Parameters
        ----------
        game_dict: dict
            A dictionary of data pertaining to an NBA game

        Returns
        -------
        Game
            A Game object that represents this particular NBA game
        """
        home_team = self.generate_team_from_team_dict(game_dict['home'])
        away_team = self.generate_team_from_team_dict(game_dict['away'])
        code: str = game_dict['code']
        season: str = game_dict['season']
        date: str = game_dict['date']
        return Game(home_team=home_team, away_team=away_team, code=code, season=season, date=date)

    def generate_player_from_player_dict(self, player_dict: Dict[Any, Any]) -> Player:
        """
        Takes a dictionary of player data and creates a Player object out of it.

        Parameters
        ----------
        player_dict: dict
            A dictionary of data that holds information about a certain Players performance in one NBA game

        Returns
        -------
        Player
            The Player object converted from the dictionary of player data
        """
        try:
            name: str = player_dict['name']
        except KeyError:
            name = ""
        position: Position = convert_to_position(player_dict['position'])
        experience: int = player_dict['experience']
        height: float = player_dict['height']
        weight: str = player_dict['weight']
        stats: Dict[PlayerStatTypes, float] = dict()
        for stat_type in player_dict:
            converted_stat_type = convert_to_player_stat_type(stat_type)
            if converted_stat_type:
                stats[converted_stat_type] = player_dict[stat_type]
        return Player(name=name, experience=experience, position=position, height=height, weight=weight, stats=stats)

    def generate_team_from_team_dict(self, team_dict: Dict[Any, Any]) -> Team:
        """
        Takes a dictionary of team data and creates a Team object out of it

        Parameters
        ----------
        team_dict: dict
            A dictionary of data that holds information about a certain Teams performance in one NBA game

        Returns
        -------
        Team
            The Team object converted from the dictionary of team data
        """
        name: str = team_dict['name']
        scores: Dict[GamePeriod, int] = dict()
        for period in team_dict['scores']:
            scores[convert_to_game_period(period)] = team_dict['scores'][period]
        totals: Dict[PlayerStatTypes, float] = dict()
        for stat_type in team_dict['totals']:
            converted_type: PlayerStatTypes = convert_to_player_stat_type(stat_type)
            if converted_type:
                totals[converted_type] = team_dict['totals'][stat_type]
        players = [self.generate_player_from_player_dict(team_dict['players'][player]) for player in
                   team_dict['players'] if
                   self.generate_player_from_player_dict(team_dict['players'][player]) is not None]
        return Team(name=name, scores=scores, team_stats=totals, players=players)

    def generate_league_object(self) -> League:
        """
        Calls other methods of this class in order to build a League object that holds all of the necessary Game,
        Player, and Team objects needed to accurately hold data for all games played in the NBA provided by the JSON
        files.

        Returns
        -------
        League
            The league object that holds all of the data for each NBA game.
        """
        seasons_dict: Dict[str, List[Dict[Any, Any]]] = dict()
        for game in self.games:
            seasons_dict.setdefault(game.season, list())
            seasons_dict[game.season].append(game)
        return League(seasons_dict)
