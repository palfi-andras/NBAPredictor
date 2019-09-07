from enum import Enum


class Position(Enum):
    PG = 1  # Point Guard
    SG = 2  # Shooting Guard
    SF = 3  # Small Forward
    PF = 4  # Power Forward
    C = 5  # Center


def convert_to_position(char: str) -> Position:
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
        raise Exception(f"Unrecognized position: {char}")
