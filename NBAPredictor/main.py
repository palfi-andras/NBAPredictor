import os

from tap import Tap

from core.league import load_league
from core.nba_json_parser import NBAJsonParser


class NBAPredictorArguments(Tap):
    dir: str = './resources/nba'  # Optional path for directory of where JSON
    # files are located when scraped from basketballrefernce.com
    rebuild: bool = False  # Pass this flag in to rebuild the library of players and previous games. Whenever the   #
    # library is built, it is saved so it can be reused again for the next execution of the program. This flag forces
    # the rebuild of the entire League and will add some time to program startup
    league_save: str = './resources/league.pkl'  # Location of the league save file (Pickle Object)


if __name__ == '__main__':
    args: NBAPredictorArguments = NBAPredictorArguments().parse_args()
    if not args.rebuild and not os.path.isfile(args.league_save):
        args.rebuild = True
    if args.rebuild:
        league = NBAJsonParser(args.dir).generate_league_object()
        print()
        league.save_league(args.league_save)
    else:
        league = load_league(args.league_save)
        league.save_league(args.league_save)
