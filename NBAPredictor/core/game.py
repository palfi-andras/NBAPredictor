from team import Team
from player_stat_types import PlayerStatTypes


class Game:

    def __init__(self, home_team: Team, away_team: Team, code: str, season: str, date: str):
        self.home_team = home_team
        self.away_team = away_team
        self.code = code
        self.season = season
        self.date = date

