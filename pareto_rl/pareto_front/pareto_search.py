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

from typing import Dict, List, Tuple
from inspyred.ec import variators
from pareto_rl.pareto_front.ga.utils.inspyred_utils import NumpyRandomWrapper
from poke_env.environment.pokemon import Pokemon
from pareto_rl.dql_agent.utils.move import Move
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.pareto_front.ga.nsga2 import nsga2
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
from pareto_rl.pareto_front.classes.next_turn import (
    NextTurn,
    next_turn_mutation,
    next_turn_crossover,
    NextTurnTest,
    next_turn_test_mutation,
)
import matplotlib.pyplot as plt
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper


def configure_subparsers(subparsers):
    r"""Configure a new subparser for testing the Pareto search.

    Args:
      subparsers: subparser
    """

    """
    Subparser parameters:
    Args:
        dry [bool]: whether to run without showing any plot
    """
    parser = subparsers.add_parser("pareto", help="Test the Pareto search")
    parser.add_argument(
        "--dry", action="store_true", default=False, help="Run without showing plots"
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


def pareto_search(
    args,
    battle: DoubleBattle = None,
    pm: PokemonMapper = None,
    player=None,
) -> List[DoubleBattleOrder]:
    r"""Main function which runs the pareto search returning the final population and final population fitness
    It can perform it either on a Showdown battle or on some static pokemon team
    Args:
        - args: command line arguments
        - battle [DoubleBattle] = None: Pokémon battle
        - pm [PokemonMapper] = None: pokemon mapper
        - last_turn [List[Tuple[str, str]]] = None: last turn
    Returns:
        - lists of [DoubleBattleOrder]
    """

    # If dry do not show any plot
    display = not args.dry

    # parameters for NSGA-2
    nsga2_args = {}
    nsga2_args["pop_size"] = 20
    nsga2_args["max_generations"] = 10

    """
    -------------------------------------------------------------------------
    """
    if (battle is not None) and (pm is not None):
        problem = NextTurn(battle, pm, player)
        # crossover and mutation
        nsga2_args["variator"] = [next_turn_crossover, next_turn_mutation]
    else:
        problem = NextTurnTest()
        # crossover and mutation
        nsga2_args["variator"] = [variators.uniform_crossover, next_turn_test_mutation]

    # name of the objective for plot purposes
    nsga2_args["objective_0"] = "Mon Dmg"
    nsga2_args["objective_1"] = "Opp Dmg"
    nsga2_args["objective_2"] = "Mon HP"
    nsga2_args["objective_3"] = "Opp HP"

    nsga2_args["fig_title"] = "Pokémon NSGA-2"

    rng = NumpyRandomWrapper()

    # runs nsga2
    final_pop, final_pop_fitnesses = nsga2(
        rng, problem, display=display, num_vars=8, **nsga2_args
    )

    if not args.dry:
        print("Final Population\n", final_pop)
        print("Final Population Fitnesses\n", final_pop_fitnesses)
        plt.ioff()
        plt.show()

    orders = []

    # Build battle orders starting from the
    # final population by the means of the
    # Pokémon mapper
    for c in final_pop:
        first_order = None
        second_order = None
        for i in range(0, len(c), 2):
            pos = pm.get_field_pos_from_genotype(i)
            action = c[i]
            target = DoubleBattle.EMPTY_TARGET_POSITION
            if isinstance(action, Move):
                target = c[i + 1] if c[i + 1] < 3 else 0
                target = 0 if action.deduced_target in ['self', 'randomNormal'] else target
            if pos < 0:
                if first_order is None:
                    first_order = BattleOrder(action, move_target=target)
                else:
                    second_order = BattleOrder(action, move_target=target)
        orders.append(
            DoubleBattleOrder(first_order=first_order, second_order=second_order)
        )

    return orders
