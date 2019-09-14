from typing import Dict, List, Any

from player_stat_types import PlayerStatTypes
from positions import Position
#from league import League

class Player:

    def __init__(self, name: str, experience: int, position: Position, height: float, weight: str,
            stats: Dict[str, List[Dict[PlayerStatTypes, float]]]):
        self.name = name
        self.experience = experience
        self.position = position
        self.height = height
        self.weight = weight
        self.stats = stats

    def add_new_stats(self, season: str, stats: Dict[PlayerStatTypes, float]) -> None:
        self.stats[season].append(stats)

    def update_experience(self, new_experience: int) -> None:
        if new_experience > self.experience:
            self.experience = new_experience

#def generate_player_object(league: League, game_dict: Dict[Any, Any])-> Player:
#    if league.players_exists():
 #       pass
