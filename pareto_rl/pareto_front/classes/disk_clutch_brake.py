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
    ['sludge bomb', 'shadow ball', 'hex', 'toxic'],                 #pokemon1.moves,     #4    # Gengar
    ['vulpix', 'charmander', 'venusaur'],                           #pokemon_on_field,   #3
    ['fire fang', 'flamethrower', 'sunny day', 'will-o-wisp'],      #pokemon2.moves,     #4    # Vulpix
    ['gengar', 'charmander', 'venusaur'],                           #pokemon_on_field,   #3
    ['fire punch', 'ember', 'earth power', 'tackle'],               #opponent1.moves,    #4    # Charmander
    ['gengar', 'vulpix', 'venusaur'],                               #pokemon_on_field,   #3
    ['energy ball', 'sleep powder', 'frenzy plant', 'leaf storm'],  #opponent2.moves,    #4    # Venusaur
    ['gengar', 'vulpix', 'charmander']                              #pokemon_on_field    #3
    #np.arange(60, 81, 1),
    #np.arange(90, 111, 1),
    #np.arange(1.5, 3.5, 0.5),
    #np.arange(600, 1010, 10),
    #np.arange(2, 10, 1),
]

class DiskClutchBrake(benchmarks.Benchmark):
    def __init__(self, constrained=False):
        # n_dimensions and n_objectives
        benchmarks.Benchmark.__init__(self, 8, 4)
        self.maximize = True
        self.constrained = constrained

    def generator(self, random, args):
        return [random.sample(values[i], 1)[0] for i in range(self.dimensions)]

    def evaluator(self, candidates, args):
        fitness = []
        request = {'requests': []}
        for i, c in enumerate(candidates):
            request['requests'].append(prepare_request(pkmn, c))

        result = damage_request(json.dumps(request))
        result = json.loads(result)

        for r in result['results']:
            attacks = []
            for attack in r.values():
                damage = attack['damage']
                attacks.append(damage.pop() if isinstance(damage, list) else damage)

            f1 = attacks[0] + attacks[1]

            f2 = attacks[2] + attacks[3]

            f3 = max((261+217) - f2, 0)

            f4 = max((301+219) - f1, 0)

            fitness.append(
                Pareto([f1, f2, f3, f4], self.maximize)
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


def prepare_request(pkmn, c):
    return {
        0: {'Attacker': pkmn[0], 'Move': c[0], 'Defender': c[1]},
        1: {'Attacker': pkmn[1], 'Move': c[2], 'Defender': c[3]},
        2: {'Attacker': pkmn[2], 'Move': c[4], 'Defender': c[5]},
        3: {'Attacker': pkmn[3], 'Move': c[6], 'Defender': c[7]}
    }