""" nsga2.py
Module which contains the NSGA-II algorithm implementation.

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

import time
from inspyred.ec.emo import NSGA2
from inspyred.ec import terminators, variators
from pareto_rl.dql_agent.utils.utils import get_run_folder

from pareto_rl.pareto_front.ga.utils import inspyred_utils
import numpy as np
import os
import shutil
import csv
import json
from pathlib import Path

# NSGA runs folder
FOLDER = f"{Path(__file__).parent.absolute()}/../../../nsga2_runs/"

def init_nsga2():
    r"""NSGA2 algorithm initialization, it clears the folder of the nsga2
    """
    for filename in os.listdir(get_run_folder(FOLDER)):
        file_path = os.path.join(get_run_folder(FOLDER), filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def nsga2(random, problem, num_vars=0, variator=None, **kwargs):
    r"""NSGA2 algorithm

    Args:
        - random: random number generator
        - problem, problem to solve
        - display=False, wether to display plots
        - num_vars=0, how many variables does the problem have
        - variator=None, crossover and mutation to apply
        - **kwargs, additional arguments such as the population size
        and the number of generations
    """

    # create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}

    algorithm = NSGA2(random)
    # algorithm terminator
    algorithm.terminator = terminators.generation_termination

    # crossover and mutation
    if variator is None:
        algorithm.variator = [variators.blend_crossover, variators.gaussian_mutation]
    else:
        algorithm.variator = variator

    # pass problem to args
    kwargs["problem"] = problem
    # population size
    kwargs["num_selected"] = kwargs["pop_size"]

    if problem.objectives == 2:
        algorithm.observer = [inspyred_utils.initial_pop_observer]
    else:
        algorithm.observer = inspyred_utils.initial_pop_observer

    # evolve
    final_pop = algorithm.evolve(
        evaluator=problem.evaluator,
        maximize=problem.maximize,
        initial_pop_storage=initial_pop_storage,
        num_vars=num_vars,
        generator=problem.generator,
        **kwargs
    )

    best_guy = final_pop[0].candidate[0:num_vars]
    best_fitness = final_pop[0].fitness
    # final_pop_fitnesses = asarray([guy.fitness for guy in algorithm.archive])
    # final_pop_candidates = asarray([guy.candidate[0:num_vars] for guy in algorithm.archive])

    final_pop_fitnesses = np.asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = np.asarray([guy.candidate[0:num_vars] for guy in final_pop])

    # dump the current population
    save_current_population(final_pop_fitnesses.tolist(), kwargs)

    return final_pop_candidates, final_pop_fitnesses

def current_millis_time():
    r"""Gets the current time millis
    """
    return round(time.time() * 10**7)

def save_current_population(population, kwargs):
    """
    Save current population values of a NSGA2 run inside of a CSV file

    Args:
        population: population fitness
        kwargs: dictinary of additional arguments
    """
    get_file_name = current_millis_time()
    folder = get_run_folder(FOLDER)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(f"{folder}{get_file_name}.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(["population","args"]) # header
        pop = json.dumps(population)
        arg = json.dumps({k: v for k, v in kwargs.items() if k.startswith('objective')})
        writer.writerow([pop,arg])

def get_evaluations(foldername):
    """
    Get all the evaluation files inside the given folder

    Args:
        foldername: foldername
    """
    visible_files = [
        file.name for file in Path(foldername).iterdir() if not file.name.startswith(".")
    ]

    files = sorted(
        filter( lambda x: os.path.isfile(os.path.join(foldername, x)), visible_files),
        key = lambda x: int(x.split('.', 1)[0])
    )

    return list(map(lambda x: f"{foldername}{x}", files))


def parse_evaluation(filename):
    """
    Parse a single evaluation from a CSV file

    Args:
        filename: filename
    """
    with open(f"{filename}") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        data = next(csv_reader)

    return json.loads(data[0]), json.loads(data[1])
