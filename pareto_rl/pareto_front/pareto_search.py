""" pareto_search.py
Module to test the Pareto front data extraction

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

from typing_extensions import final
from typing import Tuple, Optional, List
from inspyred.ec import variators
from pareto_rl.pareto_front.ga.utils.inspyred_utils import NumpyRandomWrapper
from poke_env.environment.pokemon import Pokemon
from pareto_rl.pareto_front.ga.nsga2 import nsga2
from poke_env.player.battle_order import BattleOrder
from pareto_rl.pareto_front.classes.next_turn import (
    NextTurn,
    NextTurnTest,
    next_turn_mutation,
)
import matplotlib.pyplot as plt


def configure_subparsers(subparsers):
    r"""Configure a new subparser for testing the Pareto search.

    Args:
      subparsers: subparser
    """

    """
    Subparser parameters:
    Args:
    
    """
    parser = subparsers.add_parser("pareto", help="Test the Pareto search")
    parser.add_argument(
      "--dry", action='store_true', default=False, help="Run without showing plots"
    )
    parser.set_defaults(func=main)


def main(args):
    r"""Checks the command line arguments and then runs pareto_search.
    Args:
      args: command line arguments
    """

    print("\n### Pareto search ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    pareto_search(args)


def pareto_search(args, orders: Optional[List[Tuple[BattleOrder, Pokemon]]] = None):
    r"""Main function which runs the pareto search returning the final population and final population fitness
    Args:
      args: command line arguments
    """

    # If dry do not show any plot
    display = not args.dry

    # parameters for NSGA-2
    nsga2_args = {}
    nsga2_args["pop_size"] = 50
    nsga2_args["max_generations"] = 10

    """
    -------------------------------------------------------------------------
    """
    if orders is not None:
      # Orders in DoubleBattleOrders and the respective pokémon received
      print(f"{orders[0][0]} {orders[0][1]}")
      problem = NextTurn()
    else:
      problem = NextTurnTest()

    # name of the objective for plot purpose
    nsga2_args["objective_0"] = "Mon Dmg"
    nsga2_args["objective_1"] = "Opp Dmg"
    nsga2_args["objective_2"] = "Mon HP"
    nsga2_args["objective_3"] = "Opp HP"

    # crossover and mutation
    nsga2_args["variator"] = [variators.uniform_crossover, next_turn_mutation]

    nsga2_args["fig_title"] = "Pokémon NSGA-2"

    rng = NumpyRandomWrapper()

    final_pop, final_pop_fitnesses = nsga2(
        rng, problem, display=display, num_vars=8, **nsga2_args
    )

    if not args.dry:
      print("Final Population\n", final_pop)
      print("Final Population Fitnesses\n", final_pop_fitnesses)
      plt.ioff()
      plt.show()
    
    return final_pop, final_pop_fitnesses
