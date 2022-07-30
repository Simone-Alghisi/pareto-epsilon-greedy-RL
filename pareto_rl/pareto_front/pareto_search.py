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

from typing import List
from poke_env.player.player import Player
from pareto_rl.dql_agent.utils.utils import get_run_folder
from pareto_rl.pareto_front.ga.utils.inspyred_utils import NumpyRandomWrapper
from pareto_rl.dql_agent.utils.move import Move
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.pareto_front.ga.nsga2 import (
    nsga2,
    init_nsga2,
    get_evaluations,
    parse_evaluation
)
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
from pareto_rl.pareto_front.classes.next_turn import (
    NextTurn,
    next_turn_mutation,
    next_turn_crossover,
)
import pareto_rl.pareto_front.ga.utils.plot_utils as plot_utils
import matplotlib.pyplot as plt
# from adjustText import adjust_text
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from pathlib import Path

import numpy as np
import seaborn as sb
import pandas as pd

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

    # init nsga2
    init_nsga2()

    pareto_search(args)

    # plot the diagrams
    if not args.dry:
        folder = get_run_folder(f"{Path(__file__).parent.absolute()}/../../nsga2_runs/")
        files = get_evaluations(folder)
        populations = []
        for i, file in enumerate(files):
            population, args = parse_evaluation(file)
            populations.append(population)
            plot_utils.generate_multilevel_diagram(population, f"NSGA2 run {i} multilevel diagram")
            plot_utils.plot_results_multi_objective_PF(population, f"NSGA2 run {i} objective vs objective", args)
        plot_utils.generate_multilevel_diagram_different_population(populations, "NSGA2 runs comparisons multilevel diagram")

def pareto_search(
    args,
    battle: DoubleBattle,
    pm: PokemonMapper,
    player: Player,
) -> List[DoubleBattleOrder]:
    r"""Main function which runs the pareto search returning the final population and final population fitness
    It can perform it either on a Showdown battle or on some static pokemon team
    Args:
        - args: command line arguments
        - battle [DoubleBattle] = None: Pokémon battle
        - pm [PokemonMapper] = None: pokemon mapper
    Returns:
        - lists of [DoubleBattleOrder]
    """

    # parameters for NSGA-2
    nsga2_args = {}
    nsga2_args["pop_size"] = 40
    nsga2_args["max_generations"] = 10

    """
    -------------------------------------------------------------------------
    """
    problem = NextTurn(battle, pm, player)
    # crossover and mutation
    nsga2_args["variator"] = [next_turn_crossover, next_turn_mutation]

    # name of the objective for plot purposes
    nsga2_args["objective_0"] = "Mon Dmg"
    nsga2_args["objective_1"] = "Opp Dmg"
    nsga2_args["objective_2"] = "Mon HP"
    nsga2_args["objective_3"] = "Opp HP"

    nsga2_args["fig_title"] = "Pokémon NSGA-2"

    rng = NumpyRandomWrapper()

    # runs nsga2
    final_pop, final_pop_fitnesses = nsga2(
        rng, problem, num_vars=8, **nsga2_args
    )

    if not args.dry:
        print("Final Population\n", final_pop)
        print("Final Population Fitnesses\n", final_pop_fitnesses)

    orders = []

    # Build battle orders starting from the
    # final population by the means of the pokemapper
    for c in final_pop:
        first_order = None
        second_order = None
        for i in range(0, len(c), 2):
            pos = pm.get_field_pos_from_genotype(i)
            action = c[i]
            target = DoubleBattle.EMPTY_TARGET_POSITION
            if isinstance(action, Move):
                target = c[i + 1] if c[i + 1] < 3 else 0
                target = (
                    0 if action.deduced_target in ["self", "randomNormal"] else target
                )
            if pos < 0:
                if first_order is None:
                    first_order = BattleOrder(action, move_target=target)
                else:
                    second_order = BattleOrder(action, move_target=target)
        orders.append(
            DoubleBattleOrder(first_order=first_order, second_order=second_order)
        )

    return orders
