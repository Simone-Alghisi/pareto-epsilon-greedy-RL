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

from inspyred.ec import variators
from pareto_rl.pareto_front.ga.utils.inspyred_utils import NumpyRandomWrapper
from pareto_rl.pareto_front.ga.nsga2 import nsga2
from pareto_rl.pareto_front.classes.disk_clutch_brake import (
    DiskClutchBrake,
    disk_clutch_brake_mutation,
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
    parser.set_defaults(func=main)


def main(args):
    r"""Checks the command line arguments and then runs pareto_search.
    Args:
      args: command line arguments
    """
    # check some parameters
    # TODO

    print("\n### Testing Pareto search ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    pareto_search(args)


def pareto_search(args):
    r"""Main function which ..
    Args:
      args: command line arguments
    """

    display = True

    # parameters for NSGA-2
    args = {}
    args["pop_size"] = 50
    args["max_generations"] = 10
    constrained = False

    """
  -------------------------------------------------------------------------
  """

    problem = DiskClutchBrake(constrained)
    if constrained:
        args["constraint_function"] = problem.constraint_function
    args["objective_1"] = "Brake Mass (kg)"
    args["objective_2"] = "Stopping Time (s)"

    args["variator"] = [variators.blend_crossover, disk_clutch_brake_mutation]

    args["fig_title"] = "NSGA-2"

    rng = NumpyRandomWrapper()

    final_pop, final_pop_fitnesses = nsga2(
        rng, problem, display=display, num_vars=5, **args
    )

    print("Final Population\n", final_pop)
    print("Final Population Fitnesses\n", final_pop_fitnesses)
    plt.ioff()
    plt.show()
