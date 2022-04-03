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
import json
from pareto_rl.damage_calculator.requester import damage_request

pkmn = ['gengar', 'vulpix', 'charmander', 'venusaur']

# possible values
values = [
    ['ember', 'hydropump', 'hex', 'toxic'],  #pokemon1.moves,     #4
    ['vulpix', 'charmander', 'venusaur'],    #pokemon_on_field,   #3
    ['ember', 'hydropump', 'hex', 'toxic'],  #pokemon2.moves,     #4
    ['gengar', 'charmander', 'venusaur'],    #pokemon_on_field,   #3
    ['ember', 'hydropump', 'hex', 'toxic'],  #opponent1.moves,    #4
    ['gengar', 'vulpix', 'venusaur'],        #pokemon_on_field,   #3
    ['ember', 'hydropump', 'hex', 'toxic'],  #opponent2.moves,    #4
    ['gengar', 'vulpix', 'charmander']       #pokemon_on_field    #3
    #np.arange(60, 81, 1),
    #np.arange(90, 111, 1),
    #np.arange(1.5, 3.5, 0.5),
    #np.arange(600, 1010, 10),
    #np.arange(2, 10, 1),
]

#class DiskClutchBounder(object):
#    def __call__(self, candidate, args):
#        closest = lambda target, index: min(
#            values[index], key=lambda x: abs(x - target)
#        )
#        for i, c in enumerate(candidate):
#            candidate[i] = closest(c, i)
#        return candidate

class DiskClutchBrake(benchmarks.Benchmark):
    def __init__(self, constrained=False):
        # n_dimensions and n_objectives
        benchmarks.Benchmark.__init__(self, 8, 3)
        # bounder
        # self.bounder = DiskClutchBounder()
        self.maximize = True
        self.constrained = constrained

    def generator(self, random, args):
        return [random.sample(values[i], 1)[0] for i in range(self.dimensions)]

    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:

            a1 = json.loads(damage_request(json.dumps({'Attacker': pkmn[0], 'Move': c[0], 'Defender': c[1]})))['damage']
            a1 = a1[0] if isinstance(a1, list) else a1
            a2 = json.loads(damage_request(json.dumps({'Attacker': pkmn[1], 'Move': c[2], 'Defender': c[3]})))['damage']
            a2 = a2[0] if isinstance(a2, list) else a2
            a3 = json.loads(damage_request(json.dumps({'Attacker': pkmn[2], 'Move': c[4], 'Defender': c[5]})))['damage']
            a3 = a3[0] if isinstance(a3, list) else a3
            a4 = json.loads(damage_request(json.dumps({'Attacker': pkmn[3], 'Move': c[6], 'Defender': c[7]})))['damage']
            a4 = a4[0] if isinstance(a4, list) else a4

            f1 = a1 + a2

            f2 = a3 + a4

            f3 = min((261+217) - (a3+a4),0)

            fitness.append(
                Pareto([f1, f2, f3], self.maximize)
            )

        return fitness


@mutator
def disk_clutch_brake_mutation(random, candidate, args):
    mut_rate = args.setdefault("mutation_rate", 0.1)
    bounder = args["_ec"].bounder
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < mut_rate:
            mutant[i] = random.sample(values[i], 1)[0]
    mutant = bounder(mutant, args)
    return mutant
