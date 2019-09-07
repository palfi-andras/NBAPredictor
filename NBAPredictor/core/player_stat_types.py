from enum import Enum


class PlayerStatTypes(Enum):
    MP = 1  # Minutes Played
    FG = 2  # Field Goals Made
    FGA = 3  # Field Goals Attempted
    FGP = 4  # Field Goal Percentage
    THREEP = 5  # Three Pointers Made
    THREEPA = 6  # Three Pointers Attempted
    THREEPP = 7  # Three Point Percentage
    FT = 8  # Free Throws Made
    FTA = 9  # Free Throws Attempted
    FTP = 10  # Free Throw Percentage
    ORB = 11  # Offensive Rebound
    DRB = 12  # Defensive Rebound
    TRB = 13  # Total Rebounds
    AST = 15  # Assists
    STL = 16  # Steals
    BLK = 17  # Blocks
    TOV = 18  # Turnovers
    PF = 19  # Personal Fouls
    PTS = 20  # Points
    PLM = 21  # Plus / Minus Rating
    TSP = 22  # True Shot Percentage
    EFGP = 23  # Effective FG Percentage
    THREEPAR = 24  # Three Point Attempt Rate
    FTR = 25  # Free Throw Rate
    ORBP = 26  # Offensive Rebound Percentage
    DRBP = 27  # Defensive Rebound percentage
    TRBP = 28  # Total Rebound Percentage
    ASTP = 29  # Assist Percentage
    STLP = 30  # Steal Percentage
    BLKP = 31  # Block Percentage
    TOVP = 32  # Turnover Percentage
    USG = 33  # Usage
    ORTG = 34  # Offensive Rating
    DRTF = 35  # Defensive Rating
    ASTTOV = 36  # Assist Turnover Ratio
    STLTOV = 37  # Steal Turnover Ration
    HOB = 38  # Hands On Buckets
    FIC = 39  # Floor Impact Counter


def convert_to_player_stat_type(char: str) -> PlayerStatTypes:
    if char == 'MP':
        return PlayerStatTypes.MP
    elif char == 'FG':
        return PlayerStatTypes.FG
    elif char == 'FGA':
        return PlayerStatTypes.FGA
    elif char == 'FG%':
        return PlayerStatTypes.FGP
    elif char == '3P':
        return PlayerStatTypes.THREEP
    elif char == '3PA':
        return PlayerStatTypes.THREEPA
    elif char == '3P%':
        return PlayerStatTypes.THREEPP
    elif char == 'FT':
        return PlayerStatTypes.FT
    elif char == 'FTA':
        return PlayerStatTypes.FTA
    elif char == 'FT%':
        return PlayerStatTypes.FTP
    elif char == 'ORB':
        return PlayerStatTypes.ORB
    elif char == 'DRB':
        return PlayerStatTypes.DRB
    elif char == 'TRB':
        return PlayerStatTypes.TRB
    elif char == 'AST':
        return PlayerStatTypes.AST
    elif char == 'STL':
        return PlayerStatTypes.STL
    elif char == 'BLK':
        return PlayerStatTypes.BLK
    elif char == 'TOV':
        return PlayerStatTypes.TOV
    elif char == 'PF':
        return PlayerStatTypes.PF
    elif char == 'PTS':
        return PlayerStatTypes.PTS
    elif char == '+/-':
        return PlayerStatTypes.PLM
    elif char == 'TS%':
        return PlayerStatTypes.TSP
    elif char == 'eFG%':
        return PlayerStatTypes.EFGP
    elif char == '3PAr':
        return PlayerStatTypes.THREEPAR
    elif char == 'FTr':
        return PlayerStatTypes.FTR
    elif char == 'ORB%':
        return PlayerStatTypes.ORBP
    elif char == 'DRB%':
        return PlayerStatTypes.DRBP
    elif char == 'TRB%':
        return PlayerStatTypes.TRBP
    elif char == 'AST%':
        return PlayerStatTypes.ASTP
    elif char == 'STL%':
        return PlayerStatTypes.STLP
    elif char == 'BLK%':
        return PlayerStatTypes.BLKP
    elif char == 'TOV%':
        return PlayerStatTypes.TOVP
    elif char == 'USG%':
        return PlayerStatTypes.USG
    elif char == 'ORtg':
        return PlayerStatTypes.ORTG
    elif char == 'DRtg':
        return PlayerStatTypes.DRTF
    elif char == 'AST/TOV':
        return PlayerStatTypes.ASTTOV
    elif char == 'STL/TOV':
        return PlayerStatTypes.STLTOV
    elif char == 'HOB':
        return PlayerStatTypes.HOB
    elif char == 'FIC':
        return PlayerStatTypes.FIC
    else:
        print(f"Unrecognized or non-implemented stat type: {char}")
        return None
