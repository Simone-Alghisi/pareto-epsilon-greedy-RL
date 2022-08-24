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

import asyncio
import argparse
import pareto_rl.pareto_front as pareto_processor
import pareto_rl.test as test_processor
import logging
import matplotlib
from pareto_rl.dql_agent import agent

#: mapping between string logger levels and their actual value
LOGGER_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def get_args():
    r"""Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m pareto_rl",
        description="Reinforcement learning for Pokémon battles",
    )

    # subparsers
    subparsers = parser.add_subparsers(help="sub-commands help")
    test_processor.pareto_battle.configure_subparsers(subparsers)
    agent.configure_subparsers(subparsers)

    # arguments of the parser
    parser.add_argument(
        "--logger-level",
        "-ll",
        choices=LOGGER_LEVELS.keys(),
        default="WARNING",
        help="Logger level",
    )
    parser.add_argument(
        "--matplotlib-backend",
        "-mb",
        choices=matplotlib.rcsetup.interactive_bk,
        default="QtAgg",
        help="Matplotlib interactive backend",
    )

    # parse arguments
    parsed_args = parser.parse_args()
    parsed_args.logger_level = LOGGER_LEVELS.get(parsed_args.logger_level)

    return parsed_args


def log(level):
    r"""Initialise the logger.

    Args:
        level [int]: logging level, one of {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # create file handler and set level
    ch = logging.FileHandler(filename="log/log.txt", mode="w", encoding="utf-8")
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def main(args):
    r"""Main function"""
    log(args.logger_level)
    matplotlib.use(args.matplotlib_backend)
    args.func(
        args,
    )


if __name__ == "__main__":
    main(get_args())
