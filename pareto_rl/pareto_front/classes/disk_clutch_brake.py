""" disk_clutch_brake.py
Module which contanis some util functions using the inspyred library.

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

from inspyred import benchmarks
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
import numpy as np
import copy

# parameters, see Deb 2006
Delta_R = 20  # mm
L_max = 30  # mm
delta = 0.5  # mm
p_max = 1  # MPa
V_sr_max = 10  # m/s
n = 250  # rpm
mu = 0.5
s = 1.5
M_s = 40  # Nm
omega = np.pi * n / 30.0  # rad/s
rho = 0.0000078  # kg/mm^3
T_max = 15  # s
M_f = 3  # Nm
I_z = 55  # kg*m^2

# possible values
values = [
    np.arange(60, 81, 1),
    np.arange(90, 111, 1),
    np.arange(1.5, 3.5, 0.5),
    np.arange(600, 1010, 10),
    np.arange(2, 10, 1),
]


class DiskClutchBounder(object):
    def __call__(self, candidate, args):
        closest = lambda target, index: min(
            values[index], key=lambda x: abs(x - target)
        )
        for i, c in enumerate(candidate):
            candidate[i] = closest(c, i)
        return candidate


class ConstrainedPareto(Pareto):
    def __init__(self, values=None, violations=None, ec_maximize=True):
        Pareto.__init__(self, values)
        self.violations = violations
        self.ec_maximize = ec_maximize

    def __lt__(self, other):
        if self.violations is None:
            return Pareto.__lt__(self, other)
        elif len(self.values) != len(other.values):
            raise NotImplementedError
        else:
            if self.violations > other.violations:
                # if self has more violations than other
                # return true if EC is maximizing otherwise false
                return self.ec_maximize
            elif other.violations > self.violations:
                # if other has more violations than self
                # return true if EC is minimizing otherwise false
                return not self.ec_maximize
            elif self.violations > 0:
                # if both equally infeasible (> 0) than cannot compare
                return False
            else:
                # only consider regular dominance if both are feasible
                not_worse = True
                strictly_better = False
                for x, y, m in zip(self.values, other.values, self.maximize):
                    if m:
                        if x > y:
                            not_worse = False
                        elif y > x:
                            strictly_better = True
                    else:
                        if x < y:
                            not_worse = False
                        elif y < x:
                            strictly_better = True
            return not_worse and strictly_better


class DiskClutchBrake(benchmarks.Benchmark):
    def __init__(self, constrained=False):
        benchmarks.Benchmark.__init__(self, 5, 2)
        self.bounder = DiskClutchBounder()
        self.maximize = False
        self.constrained = constrained

    def generator(self, random, args):
        return [random.sample(values[i], 1)[0] for i in range(self.dimensions)]

    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:
            f1 = np.pi * (c[1] ** 2 - c[0] ** 2) * c[2] * (c[4] + 1) * rho

            M_h = (
                (2.0 / 3.0)
                * mu
                * c[3]
                * c[4]
                * (c[1] ** 3 - c[0] ** 3)
                / (c[1] ** 2 - c[0] ** 2)
            ) / 1000.0  # N*m
            T = (I_z * omega) / (M_h + M_f)

            f2 = T

            fitness.append(
                ConstrainedPareto([f1, f2], self.constraint_function(c), self.maximize)
            )

        return fitness

    def constraint_function(self, candidate):
        if not self.constrained:
            return 0
        """Return the magnitude of constraint violations."""
        A = np.pi * (candidate[1] ** 2 - candidate[0] ** 2)  # mm^2
        p_rz = candidate[3] / A  # N/mm^2
        R_sr = (
            (2.0 / 3.0)
            * (candidate[1] ** 3 - candidate[0] ** 3)
            / (candidate[1] ** 2 - candidate[0] ** 2)
        )  # mm
        V_sr = np.pi * R_sr * n / 30000.0  # m/s

        M_h = (
            (2.0 / 3.0)
            * mu
            * candidate[3]
            * candidate[4]
            * (candidate[1] ** 3 - candidate[0] ** 3)
            / (candidate[1] ** 2 - candidate[0] ** 2)
        ) / 1000.0  # N*m

        T = (I_z * omega) / (M_h + M_f)

        violations = 0
        # g_1
        if (candidate[1] - candidate[0] - Delta_R) < 0:
            violations -= candidate[1] - candidate[0] - Delta_R
        # g_2
        if (L_max - (candidate[4] + 1) * (candidate[2] + delta)) < 0:
            violations -= L_max - (candidate[4] + 1) * (candidate[2] + delta)
        # g_3
        if (p_max - p_rz) < 0:
            violations -= p_max - p_rz
        # g_4
        if (p_max * V_sr_max - p_rz * V_sr) < 0:
            violations -= p_max * V_sr_max - p_rz * V_sr
        # g_5
        if (V_sr_max - V_sr) < 0:
            violations -= V_sr_max - V_sr
        # g_6
        if (M_h - s * M_s) < 0:
            violations -= M_h - s * M_s
        # g_7
        if T < 0:
            violations -= T
        # g_8
        if (T_max - T) < 0:
            violations -= T_max - T

        return violations


@mutator
def disk_clutch_brake_mutation(random, candidate, args):
    mut_rate = args.setdefault("mutation_rate", 0.1)
    bounder = args["_ec"].bounder
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < mut_rate:
            mutant[i] += random.gauss(0, (values[i][-1] - values[i][0]) / 10.0)
    mutant = bounder(mutant, args)
    return mutant
