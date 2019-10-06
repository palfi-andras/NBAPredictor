from enum import Enum


class Position(Enum):
    """
    An enum to represent the types of positions that are possible for Players to play in the NBA
    """
    PG = 1  # Point Guard
    SG = 2  # Shooting Guard
    SF = 3  # Small Forward
    PF = 4  # Power Forward
    C = 5  # Center


def convert_to_position(char: str) -> Position:
    """
    Converts a char into the appropriate Position, if any exists.

    Parameters
    ----------
    char: str
        A string that represents an NBA position

    Returns
    -------
    Position
        The Position represented by the string

    Raises
    ------
    RuntimeError
        If the characters cannot be converted into a Positional entry

    """
    if char == 'PG':
        return Position.PG
    elif char == 'SG':
        return Position.SG
    elif char == 'SF':
        return Position.SF
    elif char == 'PF':
        return Position.PF
    elif char == 'C':
        return Position.C
    else:
        raise RuntimeError(f"Unrecognized position: {char}")
