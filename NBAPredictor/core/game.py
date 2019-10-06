from team import Team


class Game:
    """
    A base class to represent an NBA game

    Attributes
    ----------
    home_team: Team
        A team object representing the home team
    away_team: Team
        A team object representing the away team
    code: str
        A unique code that can identify this game
    season: str
        The season the game occured in
    date: str
        The exact date of the game
    """

    def __init__(self, home_team: Team, away_team: Team, code: str, season: str, date: str):
        """
        Parameters
        ----------
        home_team: Team
            A team object representing the home team
        away_team: Team
            A team object representing the away team
        code: str
            A unique code that can identify this game
        season: str
            The season the game occured in
        date: str
            The exact date of the game
        """
        self.home_team = home_team
        self.away_team = away_team
        self.code = code
        self.season = season
        self.date = date
