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
from inspyred.ec.variators import mutator, crossover
from pareto_rl.dql_agent.utils.move import Move
from pareto_rl.damage_calculator.requester import damage_request
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from pareto_rl.dql_agent.utils.utils import get_pokemon_showdown_name
import numpy as np
import copy
import json

# possible pokemon
pkmn = ["gengar", "vulpix", "charmander", "venusaur"]

# possible values
values = [
    [
        "sludge bomb",
        "shadow ball",
        "hex",
        "toxic",
    ],  # pokemon1.moves,     #4    # Gengar
    ["vulpix", "charmander", "venusaur"],  # pokemon_on_field,   #3
    [
        "fire fang",
        "flamethrower",
        "sunny day",
        "will-o-wisp",
    ],  # pokemon2.moves,     #4    # Vulpix
    ["gengar", "charmander", "venusaur"],  # pokemon_on_field,   #3
    [
        "fire punch",
        "ember",
        "earth power",
        "tackle",
    ],  # opponent1.moves,    #4    # Charmander
    ["gengar", "vulpix", "venusaur"],  # pokemon_on_field,   #3
    [
        "energy ball",
        "sleep powder",
        "frenzy plant",
        "leaf storm",
    ],  # opponent2.moves,    #4    # Venusaur
    ["gengar", "vulpix", "charmander"],  # pokemon_on_field    #3
]


class NextTurnTest(benchmarks.Benchmark):
    def __init__(self):
        # n_dimensions and n_objectives
        benchmarks.Benchmark.__init__(self, 8, 4)
        self.maximize = True

    def generator(self, random, args):
        return [random.sample(values[i], 1)[0] for i in range(self.dimensions)]

    def evaluator(self, candidates, args):
        fitness = []
        request = {"requests": []}
        for _, c in enumerate(candidates):
            request["requests"].append(prepare_static_request(pkmn, c))

        result = damage_request(json.dumps(request))
        result = json.loads(result)

        for r in result["results"]:
            attacks = []
            for attack in r.values():
                damage = attack["damage"]
                attacks.append(damage.pop() if isinstance(damage, list) else damage)

            f1 = attacks[0] + attacks[1]

            f2 = attacks[2] + attacks[3]

            f3 = max((261 + 217) - f2, 0)

            f4 = max((301 + 219) - f1, 0)

            fitness.append(Pareto([f1, f2, f3, f4], self.maximize))
        return fitness


@mutator
def next_turn_test_mutation(random, candidate, args):
    mut_rate = args.setdefault("mutation_rate", 0.1)
    mutant = copy.copy(candidate)
    for i, _ in enumerate(mutant):
        if random.random() < mut_rate:
            mutant[i] = random.sample(values[i], 1)[0]
    return mutant


def prepare_static_request(pkmn, c):
    return {
        0: {"Attacker": pkmn[0], "Move": c[0], "Defender": c[1]},
        1: {"Attacker": pkmn[1], "Move": c[2], "Defender": c[3]},
        2: {"Attacker": pkmn[2], "Move": c[4], "Defender": c[5]},
        3: {"Attacker": pkmn[3], "Move": c[6], "Defender": c[7]},
    }

class NextTurn(benchmarks.Benchmark):
    def __init__(self, battle: DoubleBattle, pm: PokemonMapper):
        # n_dimensions and n_objectives
        benchmarks.Benchmark.__init__(self, pm.alive_pokemon_number() * 2, 4)
        self.maximize = True
        self.battle = battle
        self.pm = pm

    def generator(self, random, args):
        turn = []
        # who does what against who
        for _, moves in self.pm.moves_targets.items():
            random_move = random.choice(list(moves.keys()))
            random_target = random.choice(moves[random_move])
            turn += [random_move, random_target]
        return turn

    def evaluator(self, candidates, args):
        fitness = []
        "{-1: {moves: [-2, 2, 1]}}"
        "[{-1: {'attacker': {'name': 'mon_name', 'args': {...}}, 'target': {'name': 'mon_name', 'args': {...}}, 'move': 'move_name', 'field': {...}}, 1: {}, 2: {}}, {}, {}]"
        "[ciao, -1]"
        "get_pos_from_genotype() -> -1"
        pm = args["problem"].pm
        requests = []

        for _, c in enumerate(candidates):
            requests.append(prepare_request(c, pm))

        results = damage_request(json.dumps(requests))
        results = json.loads(results)

        for c, r in zip(candidates, results):
            mon_dmg = 0
            mon_hp = 0
            opp_dmg = 0
            opp_hp = 0

            for i in range(0, len(c), 2):
                attacker_pos = pm.get_field_pos_from_genotype(i)
                attacker = pm.pos_to_mon[attacker_pos]
                if str(attacker_pos) not in r:
                    continue
                damage = r[str(attacker_pos)]["damage"]
                if attacker_pos < 0:
                    mon_dmg += damage.pop() if isinstance(damage, list) else damage
                    mon_hp += attacker._current_hp
                else:
                    opp_dmg += damage.pop() if isinstance(damage, list) else damage
                    # find a way to discover which kind of info you know about opponent HP
                    # opp_hp = ...
                    if attacker.species.lower() == 'Zigzagoon'.lower():
                        opp_hp += 280
                    else:
                        opp_hp += 232

            mon_hp = max(mon_hp - opp_dmg, 0)
            opp_hp = max(opp_hp - mon_dmg, 0)

            fitness.append(Pareto([mon_dmg, mon_hp, opp_dmg, opp_hp], self.maximize))
        return fitness


@mutator
def next_turn_mutation(random, candidate, args):
    mut_rate = args.setdefault("mutation_rate", 0.1)
    mutant = copy.deepcopy(candidate)
    problem = args["problem"]
    pm = problem.pm
    moves_targets = pm.moves_targets
    already_mutated = False
    
    for i, _ in enumerate(candidate):
        if already_mutated:
            already_mutated = False
            continue
        if random.random() < mut_rate:
            pos = pm.get_field_pos_from_genotype(i)
            # get available moves of the mon at a certain position
            moves = moves_targets[pos]
            if i % 2 == 0:
                current_target = mutant[i + 1]
                mutated_move = random.choice(list(moves.keys()))
                mutant[i] = mutated_move
                move_targets = moves[mutated_move]
                # if target is not an option for the mutated move
                if current_target not in move_targets:
                    mutant[i + 1] = random.choice(move_targets)
                    already_mutated = True
            else:
                # pick a new target from the available ones for the current move
                move_targets = moves[mutant[i - 1]]
                mutant[i] = random.choice(move_targets)
    return mutant


@crossover
def next_turn_crossover(random, mom, dad, args):
    # TODO update the documentation (taken originally from inspyred)
    r"""Return the offspring of uniform crossover on the candidates.
    This function performs uniform crossover (UX). For each element 
    of the parents, a biased coin is flipped to determine whether 
    the first offspring gets the 'mom' or the 'dad' element. An 
    optional keyword argument in args, ``ux_bias``, determines the bias.
    .. Arguments:
       random -- the random number generator object
       mom -- the first parent candidate
       dad -- the second parent candidate
       args -- a dictionary of keyword arguments
    Optional keyword arguments in args:
    
    - *crossover_rate* -- the rate at which crossover is performed 
      (default 1.0)
    - *ux_bias* -- the bias toward the first candidate in the crossover 
      (default 0.5)
    
    """
    ux_bias = args.setdefault('ux_bias', 0.5)
    crossover_rate = args.setdefault('crossover_rate', 1.0)
    children = []
    if random.random() < crossover_rate:
        bro = copy.deepcopy(dad)
        sis = copy.deepcopy(mom)
        # perform the crossover by considering both move and target
        for i in range(0, len(dad), 2):
            if random.random() < ux_bias:
                bro[i:i+2] = mom[i:i+2]
                sis[i:i+2] = dad[i:i+2]
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children


def prepare_request(c, pm: PokemonMapper):
    request = {}
    for i in range(0, len(c), 2):
        # attacker
        attacker_pos = pm.get_field_pos_from_genotype(i)
        attacker = pm.pos_to_mon[attacker_pos]
        attacker_name = get_pokemon_showdown_name(attacker)

        # for the moment skip everything which is not a default target
        if c[i+1] > 2:
            continue

        # target
        target_pos = c[i+1]
        target = pm.pos_to_mon[target_pos]
        target_name = get_pokemon_showdown_name(target)

        # move
        move = c[i]
        move_name = move.get_showdown_name()

        request[attacker_pos] = {
            "attacker": {"name": attacker_name, "args": {}}, 
            "target": {"name": target_name, "args": {}}, 
            "move": move_name,
            "field": {}
        }
    return request