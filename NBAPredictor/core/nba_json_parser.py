"""
Class to parse NBA JSON files scraped from basketballreference.com
"""
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

    def __init__(self, nba_json_dir: str):
        self.nba_json_dir: str = nba_json_dir
        assert os.path.isdir(self.nba_json_dir), f"{self.nba_json_dir} is not a valid directory"
        self.games: List[Game] = list()
        for game_file in self.get_all_game_files():
            self.games.append(self.convert_game_dict_to_core_objects(self.parse_json_file(game_file)))

    def get_seasons(self) -> List[str]:
        valid_season_name: AnyStr = r"20[0-1][-0-9]-20[0-1][-0-9]"
        seasons: List[str] = os.listdir(self.nba_json_dir)
        for season in seasons:
            if not re.match(valid_season_name, season):
                seasons.remove(season)
        return seasons

    def get_all_game_files(self) -> List[str]:
        all_game_files: List[str] = list()
        for season in self.get_seasons():
            for game in os.listdir(os.path.join(self.nba_json_dir, season)):
                all_game_files.append(os.path.join(self.nba_json_dir, season, game))
        return all_game_files

    def parse_json_file(self, file: str) -> Dict[Any, Any]:
        assert os.path.isfile(file)
        with open(file, 'r') as file:
            json_file: Dict[Any, Any] = json.load(file)
        return json_file

    def convert_game_dict_to_core_objects(self, game_dict: Dict[Any, Any]) -> Game:
        home_team = self.generate_team_from_team_dict(game_dict['home'])
        away_team = self.generate_team_from_team_dict(game_dict['away'])
        code: str = game_dict['code']
        season: str = game_dict['season']
        date: str = game_dict['date']
        return Game(home_team=home_team, away_team=away_team, code=code, season=season, date=date)

    def generate_player_from_player_dict(self, player_dict: Dict[Any, Any]) -> Player:
        try:
            name: str = player_dict['name']
        except KeyError:
            return None
        position: Position = convert_to_position(player_dict['position'])
        experience: int = player_dict['experience']
        height: float = player_dict['height']
        weight: str = player_dict['weight']
        stats: Dict[PlayerStatTypes, float] = dict()
        for stat_type in player_dict:
            converted_stat_type: PlayerStatTypes = convert_to_player_stat_type(stat_type)
            if converted_stat_type:
                stats[converted_stat_type] = player_dict[stat_type]
        return Player(name=name, experience=experience, position=position, height=height, weight=weight, stats=stats)

    def generate_team_from_team_dict(self, team_dict: Dict[Any, Any]) -> Team:
        name: str = team_dict['name']
        scores: Dict[GamePeriod, int] = dict()
        for period in team_dict['scores']:
            scores[convert_to_game_period(period)] = team_dict['scores'][period]
        totals: Dict[PlayerStatTypes, float] = dict()
        for stat_type in team_dict['totals']:
            converted_type: PlayerStatTypes = convert_to_player_stat_type(stat_type)
            if converted_type:
                totals[converted_type] = team_dict['totals'][stat_type]
        players: List[Player] = list()
        for player in team_dict['players']:
            p = self.generate_player_from_player_dict(team_dict['players'][player])
            if p:
                players.append(p)
        return Team(name=name, scores=scores, team_stats=totals, players=players)

    def generate_league_object(self) -> League:
        seasons_dict: Dict[str, List[Dict[Any, Any]]] = dict()
        for game in self.games:
            seasons_dict.setdefault(game.season, list())
            seasons_dict[game.season].append(game)
        return League(seasons_dict)
