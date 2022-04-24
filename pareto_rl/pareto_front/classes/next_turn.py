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
from pareto_rl.damage_calculator.requester import damage_request_subprocess, damage_request_server
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from pareto_rl.dql_agent.utils.utils import get_pokemon_showdown_name, compute_opponent_stats, prepare_pokemon_request
from typing import List, Tuple, Dict, Union
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
    r"""NextTurnTest, which inherit from the benchmarks.Benchmark
    
    It is the problem class which deals with a predefined set of pokémon, 
    not attached to any Showdown pokémon battle
    """

    def __init__(self):
        r"""NexTurnTest initialization
        """
        # n_dimensions and n_objectives
        benchmarks.Benchmark.__init__(self, 8, 4)
        # whether to maximize or minimize the data
        self.maximize = True

    def generator(self, random, args):
        r"""Generator function, employed to generate a random population
        Args:
            - random: random number generator
            - args: command line arguments
        """
        return [random.sample(values[i], 1)[0] for i in range(self.dimensions)]

    def evaluator(self, candidates, args):
        r"""Evaluator function, computes the fitness for the actual population
        Args:
            - candidates: actual population
            - args: command line arguments
        """
        fitness = []
        request = {"requests": []}
        # prepare the damage request starting from the actual population
        for _, c in enumerate(candidates):
            request["requests"].append(prepare_static_request(pkmn, c))

        # ask for the damage to the damage requester
        result = damage_request_subprocess(json.dumps(request))
        result = json.loads(result)

        # compute the fitenss
        # This fitness tries to to maximize the ally pokémon damage
        # and in the meanwhile minimise the opponent damage
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
    r"""Mutation
    Args:
        - random: random number generator
        - candidate: individual
        - args: command line arguments
    """
    mut_rate = args.setdefault("mutation_rate", 0.1)
    mutant = copy.copy(candidate)
    for i, _ in enumerate(mutant):
        if random.random() < mut_rate:
            # random sample a pokémon
            mutant[i] = random.sample(values[i], 1)[0]
    return mutant


def prepare_static_request(pkmn, c):
    r"""Prepare static request method, employed in order to
    prepare a request to the damage calculator
    Args:
        - pkmn: current pokemon
        - c: pokémon array
    """
    return {
        0: {"Attacker": pkmn[0], "Move": c[0], "Defender": c[1]},
        1: {"Attacker": pkmn[1], "Move": c[2], "Defender": c[3]},
        2: {"Attacker": pkmn[2], "Move": c[4], "Defender": c[5]},
        3: {"Attacker": pkmn[3], "Move": c[6], "Defender": c[7]},
    }

class NextTurn(benchmarks.Benchmark):
    r"""NextTurn, which inherit from the benchmarks.Benchmark
    
    It is the problem class which deals with a Pokémon battle 
    attatched to one Showdown pokémon battle
    """
    def __init__(self, battle: DoubleBattle, pm: PokemonMapper, last_turn: List[Tuple[str, str]]):
        # n_dimensions and n_objectives
        benchmarks.Benchmark.__init__(self, pm.alive_pokemon_number() * 2, 4)
        self.maximize = True
        self.battle = battle
        self.pm = pm
        self.last_turn = last_turn
        # buffer for the actual turn, it stores the result of the damage calculator
        # without the need of sending a request
        self.turn_buffer = {}

    def generator(self, random, args) -> List[Union[Move, int]]:
        r"""
        Returns an initial set of individuals with a fixed
        number of genes (two for each mon) that is used to encode 
        the moves performed and the target in the current turn.
        Args:
            - random: a random generator
            - args: args passed to the instance
        Returns:
            - candidates [List]: the initial list of candidates for the
            current turn
        """
        turn = []
        # who does what against who
        for _, moves in self.pm.moves_targets.items():
            random_move = random.choice(list(moves.keys()))
            random_target = random.choice(moves[random_move])
            turn += [random_move, random_target]
        return turn

    def evaluator(self, candidates, args) -> List[Pareto]:
        r"""
        Evaluates the current set of candidates based on 4 different
        objectives (two for the allies and two for the opponents), i.e. 
        - the damage inflicted to the opponents;
        - the remaining HP. 
        For the computation of the damage, smogon damage_calculator is
        used (request to a local instance).
        Args:
            - candidates: set of candidate individuals
        Returns:
            - fitness: list containing a Pareto problem with the value
            for the 4 objectives that need to be maximized
        """
        fitness = []
        pm = args["problem"].pm
        last_turn = args["problem"].last_turn
        request = {"requests": []}

        n_mon = 0
        n_opp = 0

        a = time.time()
        starting_hp: Dict[int, int] = {}
        for pos, mon in pm.pos_to_mon.items():
            if pos < 0:
                starting_hp[pos] = mon.current_hp
                n_mon += 1
            else:
                starting_hp[pos] = mon.current_hp*compute_opponent_stats('hp', mon)/100
                n_opp += 1
        for c in candidates:

            # prepare a possible request for the candidate 
            possible_requests = prepare_request(c, pm)
            for i in range(0, len(c), 2):
                # insert the elements in the buffer
                attacker_pos = pm.get_field_pos_from_genotype(i)
                # target
                target_pos = c[i+1]
                # move
                move = c[i]
                # key of the hash
                key = hash(f"{attacker_pos}{target_pos}{move}")
                # exploit the turn buffer
                if key in self.turn_buffer:
                    r = self.turn_buffer[key]
                # TODO... handle cases with 3/4/5 .... as targets
                elif attacker_pos in possible_requests:
                    # send the request
                    request["requests"] = [{ attacker_pos: possible_requests[attacker_pos] }]
                    r = damage_request_server(request)
                    r = json.loads(r)
                    # save the result in the buffer
                    self.turn_buffer[key] = r

            ###################

            mon_dmg = 0
            mon_hp = 0
            opp_dmg = 0
            opp_hp = 0

            # retrieve the current turn order of the pokemon
            # TODO split into two 1. update estimates 2. compute current turn
            turn_order = get_turn_order(c, pm, last_turn)

            # a dictionary to decide wheter one mon has been
            # defeated and cannot attack anymore
            remaining_hp = starting_hp.copy()

            dmg_taken = {}

            for attacker_pos in turn_order:
                # that pokemon is already dead (like the neurons in ReLU)
                if remaining_hp[attacker_pos] == 0:
                    continue
                
                gene_idx = pm.mon_indexes.index(attacker_pos)*2
                target_pos = c[gene_idx+1]
                move = c[gene_idx]
                
                # handle all the various cases with different numbers other than
                # the default pokemon position for showdown 
                if str(attacker_pos) not in r:
                    continue

                damage = r[str(attacker_pos)]["damage"]
                if isinstance(damage, list):
                    damage = damage[len(damage)//2]

                damage = damage*move.accuracy
                
                if target_pos not in dmg_taken:
                    dmg_taken[target_pos] = 0

                # only if they are opponents the damage increases
                if target_pos*attacker_pos < 0:
                    dmg_taken[target_pos] += damage

                remaining_hp[target_pos] = max(0, remaining_hp[target_pos] - damage)

            for pos, hp in remaining_hp.items():
                if pos < 0:
                    mon_hp += (hp/starting_hp[pos])
                else:
                    opp_hp += (hp/starting_hp[pos])

            mon_hp = (mon_hp/n_mon)*100
            opp_hp = (opp_hp/n_opp)*100

            for pos, dmg in dmg_taken.items():
                if pos < 0:
                    opp_dmg += min(1, (dmg/starting_hp[pos]))
                else:
                    mon_dmg += min(1, (dmg/starting_hp[pos]))
            
            mon_dmg = (mon_dmg/n_opp)*100
            opp_dmg = (opp_dmg/n_mon)*100

            fitness.append(Pareto([mon_dmg, mon_hp, opp_dmg, opp_hp], self.maximize))
        return fitness


@mutator
def next_turn_mutation(random, candidate, args):
    r"""
    NextTurn mutation function.
    The mutation consists in choosing for the individual a new random move, which
    must be performed on a valid target. 
    If the newly defined move cannot be perfomed on the target of the old one, then a 
    new target is defined.
    If that is not possible, then the move is not changed. In such way, we are able to 
    perform a mutation operator without affecting the validity of the individual.

    Args:
        - random: random number generator
        - candidate: candidate individual
        - args: inspyred parameter.
    Returns:
        - mutant: mutated individual (if mutated)
    """
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
    r"""Return the offspring of uniform crossover on the candidates.
    This function performs uniform crossover (UX). Every two elements 
    of the parents, a biased coin is flipped to determine whether 
    the first offspring gets the 'mom' or the 'dad' element. An 
    optional keyword argument in args, ``ux_bias``, determines the bias.
    Args:
       random -- the random number generator object
       mom -- the first parent candidate
       dad -- the second parent candidate
       args -- a dictionary of keyword arguments
    Optional keyword arguments in args:
    
    - *crossover_rate* -- the rate at which crossover is performed 
      (default 1.0)
    - *ux_bias* -- the bias toward the first candidate in the crossover 
      (default 0.5)

    Returns:
    - children
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


def prepare_request(c, pm: PokemonMapper) -> Dict[int, Dict[str, Dict[str, any]]]:
    r"""
    Function which prepares the requests to send to the damage calculator
    server in order to evaluate the individual fitness

    Args:
    - c: candidate individual
    - pm [PokemonMapper]
    
    Returns:
    - requests [Dict]: request ready to be sent.
    """
    request = {}
    # for each of the pokemon in the individual (at most 4)
    for i in range(0, len(c), 2):
        # attacker
        attacker_pos = pm.get_field_pos_from_genotype(i)
        attacker = pm.pos_to_mon[attacker_pos]

        # for the moment skip everything which is not a default target
        if c[i+1] > 2:
            continue

        # target
        target_pos = c[i+1]
        target = pm.pos_to_mon[target_pos]

        # move
        move = c[i]
        move_name = move.get_showdown_name()

        attacker_args = prepare_pokemon_request(attacker)
        target_args = prepare_pokemon_request(target)
        request[attacker_pos] = {
            "attacker": attacker_args, 
            "target": target_args, 
            "move": move_name,
            "field": {}
        }
    return request


def get_turn_order(c, pm: PokemonMapper, last_turn = None) -> List[int]:
    r"""
    Returns a possible turn order (prediction) based on the current moves
    to be perfomed in the genotype for the current turn and the estimated
    speed.
    Args:
        - c: a candidate (genotype) encoding moves and target for each
        attacker
        - pm [PokemonMapper]
        - last_turn: the last turn which has been performed (TODO... maybe
        move it in a separate function)
    Returns:
        turn_order [List[int]]: a list encoding the possible turn order 
        containing the position of the attacker that will act (from first 
        to last)
    """
    turn_order: List[Tuple[int, int, int]] = []

    map_showdown_to_pos = {
        'p1a': -1,
        'p1b': -2,
        'p2a': 1,
        'p2b': 2
    }

    # This does not probably cover every case, but it's something
    if last_turn is not None and len(last_turn) > 0:
        actual_turn = []
        predicted_turn = []

        # for the moment, do not consider previous turn
        for mon_str, move_str in last_turn:
            # convert from showdown to pokemon instance
            showdown_pos = mon_str.split(':')[0]
            pos = map_showdown_to_pos[showdown_pos]
            # handle if a mon died
            if pos not in pm.pos_to_mon:
                continue
            mon = pm.pos_to_mon[pos]
            if mon.stats["spe"] is not None:
                mon_speed = mon.stats["spe"]
            else:
                mon_speed = compute_opponent_stats("spe", mon)
            move_id = Move.retrieve_id(move_str)
            move = Move(move_id)
            move_priority = move.priority
            actual_turn.append((pos, mon, move_priority, mon_speed))
            predicted_turn.append((pos, mon, move_priority, mon_speed))

        predicted_turn.sort(key=lambda x: (x[2], x[3]), reverse=True)

        already_examined = set()
        i = 0
        j = 0

        # updating the beliefs concerning the opponent pokemon speed/priority
        while i < len(actual_turn) and j < len(predicted_turn):
            pos, mon, move_priority, mon_speed = actual_turn[i]
            pr_pos, pr_mon, pr_move_priority, pr_mon_speed = predicted_turn[j]

            if pr_pos in already_examined:
                # we already adjusted the speed of that pokemon
                j += 1
            elif pos != pr_pos:
                if move_priority == pr_move_priority:
                    # at least one is an opponent, so I do not know everything about him
                    if pos > 0:
                        if "stats" not in mon._last_request:
                            mon._last_request["stats"] = {"atk": None, "def": None, "spa": None, "spd": None, "spe": None}
                        # consider case where mon it's faster
                        mon.stats["spe"] = pr_mon_speed + 1
                        print(f"{mon} should be faster, increasing speed up to {mon.stats['spe']}")
                    elif pos < 0 and pr_pos > 0:
                        if "stats" not in pr_mon._last_request:
                            mon._last_request["stats"] = {"atk": None, "def": None, "spa": None, "spd": None, "spe": None}
                        pr_mon.stats["spe"] = mon.stats["spe"]
                        print(f"{pr_mon} should be slower, so I probably got it wrong previously")
                already_examined.add(pos)
                i += 1
            else:
                # we predicted correctly
                i += 1
                j += 1

    # build the turn orders according to the priorities.
    for i in range(0, len(c), 2):
        pos = pm.get_field_pos_from_genotype(i)
        mon = pm.pos_to_mon[pos]
        if mon.stats["spe"] is not None:
            mon_speed = mon.stats["spe"]
        else:
            mon_speed = compute_opponent_stats("spe", mon)
        move = c[i]
        move_priority = move.priority
        turn_order.append((pos, move_priority, mon_speed))
    
    # sort the moves based first on their priority and then the speed of the pokemons
    turn_order.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in turn_order]