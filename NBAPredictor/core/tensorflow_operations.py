import tensorflow as tf
import os
import re
import json
from typing import Dict, Any, List, AnyStr

class TensorflowOperations:

    def __init__(self, nba_json_dir: str):
        self.nba_json_dir: str = nba_json_dir
        assert os.path.isdir(self.nba_json_dir), f"{self.nba_json_dir} is not a valid directory"

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

    def generate_datasets(self) -> List[tf.data.Dataset]:
        datasets: List[tf.data.Dataset] = list()
        for game in self.get_all_game_files():
            self.generate_dataset_from_json(game)
        return datasets

    def generate_dataset_from_json(self, json_path: str) -> tf.data.Dataset:
        assert os.path.isfile(json_path), f"Cant find file {json_path}"
        with open(json_path, 'r') as json_file:
            json_data: Dict[Any, Any] = json.load(json_file)
            print()