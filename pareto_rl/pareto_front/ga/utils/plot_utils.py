""" plot_utils.py
Module to deal with graphical representation of the data. 

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from pareto_rl.pareto_front.ga.utils.inspyred_utils import (
    CombinedObjectives,
    single_objective_evaluator,
)
import inspyred.ec.analysis


def plot_1D(axis, problem, x_limits):
    dx = (x_limits[1] - x_limits[0]) / 200.0
    x = np.arange(x_limits[0], x_limits[1] + dx, dx)
    x = x.reshape(len(x), 1)
    y = problem.evaluator(x, None)
    axis.plot(x, y, "-b")


def plot_2D(axis, problem, x_limits):
    dx = (x_limits[1] - x_limits[0]) / 50.0
    x = np.arange(x_limits[0], x_limits[1] + dx, dx)
    z = np.asarray([problem.evaluator([[i, j] for i in x], None) for j in x])
    return axis.contourf(x, x, z, 64, cmap=cm.hot_r)


def plot_results_1D(
    problem,
    individuals_1,
    fitnesses_1,
    individuals_2,
    fitnesses_2,
    title_1,
    title_2,
    args,
):
    fig = plt.figure(args["fig_title"] + " (initial and final population)")
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(individuals_1, fitnesses_1, ".b", markersize=7)
    lim = max(np.array(list(map(abs, ax1.get_xlim()))))

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(individuals_2, fitnesses_2, ".b", markersize=7)
    lim = max([lim] + np.array(list(map(abs, ax2.get_xlim()))))

    ax1.set_xlim(-lim, lim)
    ax2.set_xlim(-lim, lim)

    plot_1D(ax1, problem, [-lim, lim])
    plot_1D(ax2, problem, [-lim, lim])

    ax1.set_ylabel("Fitness")
    ax2.set_ylabel("Fitness")
    ax1.set_title(title_1)
    ax2.set_title(title_2)


def plot_results_2D(problem, individuals_1, individuals_2, title_1, title_2, args):
    fig = plt.figure(args["fig_title"] + " (initial and final population)")
    ax1 = fig.add_subplot(2, 1, 1, aspect="equal")
    ax1.plot(individuals_1[:, 0], individuals_1[:, 1], ".b", markersize=7)
    lim = max(
        np.array(list(map(abs, ax1.get_xlim())))
        + np.array(list(map(abs, ax1.get_ylim())))
    )

    ax2 = fig.add_subplot(2, 1, 2, aspect="equal")
    ax2.plot(individuals_2[:, 0], individuals_2[:, 1], ".b", markersize=7)
    lim = max(
        [lim]
        + np.array(list(map(abs, ax2.get_xlim())))
        + np.array(list(map(abs, ax2.get_ylim())))
    )

    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_title(title_1)
    ax1.locator_params(nbins=5)

    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_title(title_2)
    ax2.set_xlabel("x0")
    ax2.set_ylabel("x1")
    ax2.locator_params(nbins=5)

    plot_2D(ax1, problem, [-lim, lim])
    c = plot_2D(ax2, problem, [-lim, lim])
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    colorbar_ = plt.colorbar(c, cax=cax)
    colorbar_.ax.set_ylabel("Fitness")


"""
    multi-objective plotting utils
"""


def plot_multi_objective_1D(axis, problem, x_limits, objective):
    dx = (x_limits[1] - x_limits[0]) / 200.0
    x = np.arange(x_limits[0], x_limits[1] + dx, dx)
    x = x.reshape(len(x), 1)
    y = [f[objective] for f in problem.evaluator(x, None)]
    axis.plot(x, y, "-b")


def plot_multi_objective_2D(axis, problem, x_limits, objective):
    dx = (x_limits[1] - x_limits[0]) / 50.0
    x = np.arange(x_limits[0], x_limits[1] + dx, dx)
    z = np.asarray([problem.evaluator([[i, j] for i in x], None) for j in x])[
        :, :, int(objective)
    ]

    return axis.contourf(x, x, z, 64, cmap=cm.hot_r)


def plot_results_multi_objective_1D(
    problem,
    individuals_1,
    fitnesses_1,
    individuals_2,
    fitnesses_2,
    title_1,
    title_2,
    num_objectives,
    args,
):
    fig = plt.figure(args["fig_title"] + " (initial and final population)")
    lim = None
    axes = []
    for objective in range(num_objectives):
        ax1 = fig.add_subplot(num_objectives, 2, 2 * objective + 1)
        ax1.plot(individuals_1, [f[objective] for f in fitnesses_1], ".b", markersize=7)
        if lim is None:
            lim = max(list(map(abs, ax1.get_xlim())))
        else:
            lim = max([lim] + list(map(abs, ax1.get_xlim())))

        ax2 = fig.add_subplot(num_objectives, 2, 2 * objective + 2)
        ax2.plot(individuals_2, [f[objective] for f in fitnesses_2], ".b", markersize=7)
        lim = max([lim] + list(map(abs, ax2.get_xlim())))
        axes.append(ax1)
        axes.append(ax2)
        ax1.set_title(title_1)
        ax2.set_title(title_2)
        ax1.set_ylabel("Objective " + str(objective + 1))
        ax2.set_ylabel("Objective " + str(objective + 1))

    for i, ax in enumerate(axes):
        ax.set_xlim(-lim, lim)
        plot_multi_objective_1D(ax, problem, [-lim, lim], i / 2)


def plot_results_multi_objective_2D(
    problem, individuals_1, individuals_2, title_1, title_2, num_objectives, args
):
    fig = plt.figure(args["fig_title"] + " (initial and final population)")
    lim = None
    axes = []
    for objective in range(num_objectives):
        ax1 = fig.add_subplot(num_objectives, 2, 2 * objective + 1, aspect="equal")
        ax1.plot(individuals_1[:, 0], individuals_1[:, 1], ".b", markersize=7)
        if lim is None:
            lim = max(list(map(abs, ax1.get_xlim())) + list(map(abs, ax1.get_ylim())))
        else:
            lim = max(
                [lim] + list(map(abs, ax1.get_xlim())) + list(map(abs, ax1.get_ylim()))
            )

        ax2 = fig.add_subplot(num_objectives, 2, 2 * objective + 2, aspect="equal")
        ax2.plot(individuals_2[:, 0], individuals_2[:, 1], ".b", markersize=7)
        lim = max(
            [lim] + list(map(abs, ax2.get_xlim())) + list(map(abs, ax2.get_ylim()))
        )
        ax1.set_title(title_1)
        ax2.set_title(title_2)
        axes.append(ax1)
        axes.append(ax2)

    for i, ax in enumerate(axes):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.locator_params(nbins=5)
        objective = i / 2
        c = plot_multi_objective_2D(ax, problem, [-lim, lim], objective)
        if i % 2 == 0:
            cax = fig.add_axes(
                [
                    0.85,
                    (num_objectives - objective - 1) * (0.85 / num_objectives) + 0.12,
                    0.05,
                    0.6 / num_objectives,
                ]
            )
            colorbar_ = plt.colorbar(c, cax=cax)
            colorbar_.ax.set_ylabel("Objective " + str(objective + 1))

    fig.subplots_adjust(right=0.8)


def plot_results_multi_objective_PF(individuals, title, args):
    num_objectives = len(individuals[0].fitness)

    if num_objectives < 2:
        pass
    elif num_objectives == 2:
        plt.figure(title)
        plt.plot(
            [guy.fitness[0] for guy in individuals],
            [guy.fitness[1] for guy in individuals],
            ".b",
            markersize=7,
        )
        plt.xlabel(args["objective_0"])
        plt.ylabel(args["objective_1"])
    else:
        # Creates two subplots and unpacks the output array immediately
        f, axes = plt.subplots(
            num_objectives, num_objectives, sharex="col", sharey="row"
        )
        f.suptitle(title)
        for i in range(num_objectives):
            for j in range(num_objectives):
                axes[i, j].plot(
                    [guy.fitness[j] for guy in individuals],
                    [guy.fitness[i] for guy in individuals],
                    ".b",
                    markersize=7,
                )
                axes[i, j].set_xlabel(args["objective_" + str(j)])
                axes[i, j].set_ylabel(args["objective_" + str(i)])
        f.subplots_adjust(hspace=0.30)
        f.subplots_adjust(wspace=0.30)


"""
    the original plot_observer
"""


def plot_observer(population, num_generations, num_evaluations, args):
    """Plot the output of the evolutionary computation as a graph.

    This function plots the performance of the EC as a line graph
    using matplotlib and numpy. The graph consists of a blue line
    representing the best fitness, a green line representing the
    average fitness, and a red line representing the median fitness.
    It modifies the keyword arguments variable 'args' by including an
    entry called 'plot_data'.

    If this observer is used, the calling script should also import
    the matplotlib library and should end the script with:

    matplotlib.pyplot.show()

    Otherwise, the program may generate a runtime error.

    .. note:

    This function makes use of the matplotlib and numpy libraries.

    .. Arguments:

    population -- the population of Individuals
    num_generations -- the number of elapsed generations
    num_evaluations -- the number of candidate solution evaluations
    args -- a dictionary of keyword arguments

    """

    stats = inspyred.ec.analysis.fitness_statistics(population)
    best_fitness = stats["best"]
    worst_fitness = stats["worst"]
    median_fitness = stats["median"]
    average_fitness = stats["mean"]

    if isinstance(population[0].fitness, CombinedObjectives):
        candidates = [guy.candidate for guy in population]
        fitnesses = [
            guy.fitness for guy in single_objective_evaluator(candidates, args)
        ]
        problem = args["problem"]
        if problem.maximize:
            best_fitness = max(fitnesses)
            worst_fitness = min(fitnesses)
        else:
            best_fitness = min(fitnesses)
            worst_fitness = max(fitnesses)
        median_fitness = np.median(fitnesses)
        average_fitness = np.mean(fitnesses)

    colors = ["black", "blue", "green", "red"]
    labels = ["average", "median", "best", "worst"]
    data = []
    if num_generations == 0:
        plt.figure(args["fig_title"] + " (fitness trend)")
        plt.ion()
        data = [
            [num_evaluations],
            [average_fitness],
            [median_fitness],
            [best_fitness],
            [worst_fitness],
        ]
        lines = []
        for i in range(4):
            (line,) = plt.plot(data[0], data[i + 1], color=colors[i], label=labels[i])
            lines.append(line)
        args["plot_data"] = data
        args["plot_lines"] = lines
        plt.xlabel("Evaluations")
        plt.ylabel("Fitness")
    else:
        data = args["plot_data"]
        data[0].append(num_evaluations)
        data[1].append(average_fitness)
        data[2].append(median_fitness)
        data[3].append(best_fitness)
        data[4].append(worst_fitness)
        lines = args["plot_lines"]
        for i, line in enumerate(lines):
            line.set_xdata(np.numpy.array(data[0]))
            line.set_ydata(np.numpy.array(data[i + 1]))
        args["plot_data"] = data
        args["plot_lines"] = lines
    ymin = min([min(d) for d in data[1:]])
    ymax = max([max(d) for d in data[1:]])
    yrange = ymax - ymin
    plt.xlim((0, num_evaluations))
    plt.ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
    plt.draw()
    plt.legend()
    plt.show()
