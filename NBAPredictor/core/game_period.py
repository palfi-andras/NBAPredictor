from enum import Enum


class GamePeriod(Enum):
    FIRST_QUARTER = 1
    SECOND_QUARTER = 2
    THIRD_QUARTER = 3
    FOURTH_QUARTER = 4
    TOTAL = 5
    OT = 6
    DOT = 7
    TOT = 8
    QOT = 9


def convert_to_game_period(char: str) -> GamePeriod:
    if char == '1':
        return GamePeriod.FIRST_QUARTER
    elif char == '2':
        return GamePeriod.SECOND_QUARTER
    elif char == '3':
        return GamePeriod.THIRD_QUARTER
    elif char == '4':
        return GamePeriod.FOURTH_QUARTER
    elif char == 'T':
        return GamePeriod.TOTAL
    elif char == 'OT':
        return GamePeriod.OT
    elif char == '2OT':
        return GamePeriod.DOT
    elif char == '3OT':
        return GamePeriod.TOT
    elif char == '4OT':
        return GamePeriod.QOT
    else:
        raise Exception(f"Unrecognized char: {char}")
