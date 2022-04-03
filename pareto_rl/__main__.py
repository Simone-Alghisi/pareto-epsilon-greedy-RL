""" __main__.py
Main module that parses command line arguments.

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.

Authors:
- Simone Alghisi (simone.alghisi-1@studenti.unitn.it)
- Samuele Bortolotti (samuele.bortolotti@studenti.unitn.it)
- Massimo Rizzoli (massimo.rizzoli@studenti.unitn.it)
- Erich Robbi (erich.robbi@studenti.unitn.it)
"""

import argparse
import pareto_rl.pareto_front as processors
from pareto_rl.dql_agent import agent


def get_args():
    r"""Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m pareto_rl",
        description="Reinforcement learning for Pokémon battles",
    )

    # subparsers
    subparsers = parser.add_subparsers(help="sub-commands help")
    processors.pareto_search.configure_subparsers(subparsers)
    agent.configure_subparsers(subparsers)

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    r"""Main function."""
    args.func(
        args,
    )


if __name__ == "__main__":
    main(get_args())
