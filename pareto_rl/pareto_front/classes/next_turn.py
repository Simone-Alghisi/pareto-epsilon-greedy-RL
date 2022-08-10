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
from pareto_rl.damage_calculator.requester import (
    damage_request_server,
)
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.pokemon import Pokemon
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from pareto_rl.dql_agent.utils.utils import (
    compute_opponent_stats,
    prepare_pokemon_request,
)
from typing import Any, List, OrderedDict, Tuple, Dict, Union
import copy
import json
from random import sample
from copy import deepcopy


class NextTurn(benchmarks.Benchmark):
    r"""NextTurn, which inherit from the benchmarks.Benchmark

    It is the problem class which deals with a Pokémon battle
    attached to one Showdown pokémon battle
    """

    def __init__(self, battle: DoubleBattle, pm: PokemonMapper, player):
        # n_dimensions and n_objectives
        benchmarks.Benchmark.__init__(self, pm.alive_pokemon_number() * 2, 4)
        self.maximize = True
        self.battle = battle
        self.pm = pm
        self.player = player
        # buffer for the actual turn, it stores the result of the damage calculator
        # without the need of sending a request
        self.turn_buffer = {}
        self.p_switch = 0.05

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
        # for _, moves in self.pm.moves_targets.items():
        available_switches = deepcopy(self.pm.available_switches)
        for pos in self.pm.pos_to_mon.keys():
            if len(available_switches[pos]) > 0:
                if random.random() < self.p_switch:
                    turn += [
                        self._choose_random_switch(random, available_switches, pos),
                        None,
                    ]
                else:
                    turn += self._choose_random_move(random, pos)
            else:
                turn += self._choose_random_move(random, pos)
        return turn

    def _choose_random_switch(self, random, available_switches, pos) -> Pokemon:
        switch = random.choice(available_switches[pos])
        self._remove_inconsistent_switches(available_switches, pos, switch)
        return switch

    def _remove_inconsistent_switches(self, available_switches, pos, switch) -> None:
        ally_pos = (abs(pos) % 2 + 1) * (pos / abs(pos))
        if ally_pos in available_switches:
            for i, mon in enumerate(available_switches[ally_pos]):
                if mon.species == switch.species:
                    del available_switches[ally_pos][i]
                    break

    def _add_consistent_switches(self, available_switches, pos, switch) -> None:
        ally_pos = (abs(pos) % 2 + 1) * (pos / abs(pos))
        if ally_pos in available_switches:
            available_switches[ally_pos].append(switch)

    def _choose_random_move(self, random, pos) -> List[Union[Move, int]]:
        random_move = random.choice(list(self.pm.moves_targets[pos].keys()))
        random_target = random.choice(self.pm.moves_targets[pos][random_move])
        return [random_move, random_target]

    def _repair_switches(self, random, pos, child) -> None:
        if pos in self.pm.available_switches:
            ally_pos = int((abs(pos) % 2 + 1) * (pos / abs(pos)))
            if ally_pos in self.pm.available_switches:
                idx = self.pm.get_gene_idx_from_field_pos(pos)
                ally_idx = self.pm.get_gene_idx_from_field_pos(ally_pos)
                if (
                    isinstance(child[idx], Pokemon)
                    and isinstance(child[ally_idx], Pokemon)
                    and child[idx].species == child[ally_idx].species
                ):
                    available_switches = deepcopy(self.pm.available_switches)
                    switch = child[idx]
                    self._remove_inconsistent_switches(available_switches, pos, switch)
                    self._remove_inconsistent_switches(available_switches, ally_pos, switch)
                    if random.random() < 0.5:
                        moves = self.pm.moves_targets[pos]
                        mutate(random, available_switches, pos, child, self, moves, idx)
                    else:
                        moves = self.pm.moves_targets[ally_pos]
                        mutate(
                            random,
                            available_switches,
                            ally_pos,
                            child,
                            self,
                            moves,
                            ally_idx,
                        )

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
        pm = self.pm
        player = self.player
        data = {"requests": []}

        turns = []
        responses = {}
        requests = {}
        mapping = {}
        i = 0

        for c in candidates:
            # compute the turn order to handle correctly switches
            turn_order = get_turn_order(c, pm, player)

            # prepare the request and change who is on the field
            attacks, pos_to_mon = prepare_request(c, pm, turn_order)
            turns.append((attacks, turn_order, pos_to_mon))
            for attacker_pos, targets in attacks.items():
                for target_pos, r in targets.items():
                    move = Move(Move.retrieve_id(r["move"]))
                    key = hash(f"{attacker_pos}{target_pos}{move}")

                    # exploit the turn buffer
                    if key not in self.turn_buffer:
                        mapping[i] = key
                        requests[i] = r
                        i += 1

        # send the request
        if i > 0:
            data["requests"] = [requests]
            responses = damage_request_server(data)
            responses = json.loads(responses)
            responses = responses.pop()

            for i, response in responses.items():
                # save the result in the buffer
                key = mapping[int(i)]
                self.turn_buffer[key] = response

        for attacks, turn_order, pos_to_mon in turns:
            mon_dmg = 0
            mon_hp = 0
            opp_dmg = 0
            opp_hp = 0
            n_mon = 0
            n_opp = 0

            # Runtime hp instance, given that switches may change the pokemon on the field
            starting_hp: Dict[int, int] = {}
            for pos, mon in pos_to_mon.items():
                if pos < 0:
                    starting_hp[pos] = mon.current_hp
                    n_mon += 1
                else:
                    starting_hp[pos] = (
                        mon.current_hp * compute_opponent_stats("hp", mon)
                    ) // 100
                    n_opp += 1

            # a dictionary to decide whether one mon has been
            # defeated and cannot attack anymore
            remaining_hp = starting_hp.copy()

            dmg_taken = {}

            for attacker_pos in turn_order:
                # it's a switch, which is already been performed
                if attacker_pos not in attacks:
                    continue
                # that pokemon is already dead (like the neurons in ReLU)
                if remaining_hp[attacker_pos] == 0:
                    continue

                for target_pos, r in attacks[attacker_pos].items():
                    move = Move(Move.retrieve_id(r["move"]))
                    key = hash(f"{attacker_pos}{target_pos}{move}")

                    damage = self.turn_buffer[key]["damage"]

                    if isinstance(damage, list):
                        damage = damage[len(damage) // 2]

                    damage = damage * move.accuracy
                    if move.deduced_target == "randomNormal":
                        # if the target is random, compute the expected value (i.e. div n_opp)
                        damage = damage / len(
                            [
                                t
                                for t in pm.moves_targets.keys()
                                if (t * attacker_pos < 0)
                            ]
                        )

                    if target_pos not in dmg_taken:
                        dmg_taken[target_pos] = 0

                    # only if they are opponents the damage increases
                    if target_pos * attacker_pos < 0:
                        dmg_taken[target_pos] += damage

                    remaining_hp[target_pos] = max(0, remaining_hp[target_pos] - damage)

            for pos, hp in remaining_hp.items():
                if pos < 0:
                    mon_hp += hp / starting_hp[pos]
                else:
                    opp_hp += hp / starting_hp[pos]

            mon_hp = (mon_hp / n_mon) * 100
            opp_hp = (opp_hp / n_opp) * 100

            for pos, dmg in dmg_taken.items():
                if pos < 0:
                    opp_dmg += min(1, (dmg / starting_hp[pos]))
                else:
                    mon_dmg += min(1, (dmg / starting_hp[pos]))

            mon_dmg = (mon_dmg / n_opp) * 100
            opp_dmg = (opp_dmg / n_mon) * 100

            fitness.append(Pareto([mon_dmg, mon_hp, opp_dmg, opp_hp], self.maximize))
        return fitness


@mutator
def next_turn_mutation(random, candidate, args):
    r"""
    NextTurn mutation function.
    The mutation consists in choosing for the individual a new random move, which
    must be performed on a valid target.
    If the newly defined move cannot be performed on the target of the old one, then a
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
    problem: NextTurn = args["problem"]
    pm: PokemonMapper = problem.pm
    moves_targets = pm.moves_targets
    already_mutated = False
    available_switches = deepcopy(pm.available_switches)

    for i in range(0, len(candidate), 2):
        if isinstance(candidate[i], Pokemon):
            pos = pm.get_field_pos_from_genotype(i)
            switch = candidate[i]
            problem._remove_inconsistent_switches(available_switches, pos, switch)

    for i in range(0, len(candidate)):
        if already_mutated:
            already_mutated = False
            continue
        if random.random() < mut_rate:
            pos = pm.get_field_pos_from_genotype(i)
            # get available moves of the mon at a certain position
            moves = moves_targets[pos]
            if i % 2 == 0:
                already_mutated = mutate(
                    random, available_switches, pos, mutant, problem, moves, i
                )
            else:
                if mutant[i] is not None:
                    # pick a new target from the available ones for the current move
                    move_targets = moves[mutant[i - 1]]
                    mutant[i] = random.choice(move_targets)
    return mutant


def mutate(random, available_switches, pos, mutant, problem, moves, idx) -> bool:
    already_mutated = False
    if len(available_switches[pos]) > 0 and random.random() < problem.p_switch:
        if isinstance(mutant[idx], Pokemon):
            old_switch = mutant[idx]
            problem._add_consistent_switches(available_switches, pos, old_switch)
        mutated_switch = random.choice(available_switches[pos])
        mutant[idx] = mutated_switch
        mutant[idx + 1] = None
        problem._remove_inconsistent_switches(available_switches, pos, mutated_switch)
    else:
        mutated_move = random.choice(list(moves.keys()))
        mutant[idx] = mutated_move
        move_targets = moves[mutated_move]
        current_target = mutant[idx + 1]
        # if target is not an option for the mutated move
        if current_target is None or current_target not in move_targets:
            mutant[idx + 1] = random.choice(move_targets)
            already_mutated = True
    return already_mutated


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
    ux_bias = args.setdefault("ux_bias", 0.5)
    crossover_rate = args.setdefault("crossover_rate", 1.0)
    problem: NextTurn = args["problem"]
    children = []
    if random.random() < crossover_rate:
        bro = copy.deepcopy(dad)
        sis = copy.deepcopy(mom)
        # perform the crossover by considering both move and target
        for i in range(0, len(dad), 2):
            if random.random() < ux_bias:
                bro[i : i + 2] = mom[i : i + 2]
                sis[i : i + 2] = dad[i : i + 2]

        for child in [bro, sis]:
            for pos in [-1, 1]:
                problem._repair_switches(random, pos, child)

        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children


def prepare_request(
    c, pm: PokemonMapper, turn_order: List[int]
) -> Tuple[Dict[int, Dict[str, Dict[str, Any]]], OrderedDict[int, Pokemon]]:
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
    pos_to_mon: OrderedDict[int, Pokemon] = pm.pos_to_mon.copy()

    # for each of the pokemon in turn order
    for pos in turn_order:
        i = pm.get_gene_idx_from_field_pos(pos)
        # check if we are talking about a move
        if isinstance(c[i], Move):
            # attacker
            attacker_pos = pos
            attacker = pos_to_mon[attacker_pos]

            # target
            target_pos = c[i + 1]
            possible_targets = map_abstract_target(target_pos, attacker_pos, pm)

            # move
            move = c[i]
            move_name = move.get_showdown_name()

            attacker_args = prepare_pokemon_request(attacker, attacker_pos)
            request[attacker_pos] = {}

            for target_pos in possible_targets:
                target = pos_to_mon[target_pos]
                target_args = prepare_pokemon_request(target, target_pos)
                request[attacker_pos][target_pos] = {
                    "attacker": attacker_args,
                    "target": target_args,
                    "move": move_name,
                    # TODO map poke_env var for damage_calc
                    "field": {"gameType": "Doubles"},
                }
        elif isinstance(c[i], Pokemon):
            switch_pos = pos
            switch: Pokemon = c[i]
            # the switch is already valid
            if switch_pos < 0:
                # If we want to switch into our pokemon, we have it no matter what
                for mon in pm.battle.team.values():
                    if mon.species == switch.species:
                        pos_to_mon[switch_pos] = mon
                        break
            else:
                found = False
                # If we are considering an opponent pokemon, we may have to create it
                for mon in pm.battle.opponent_team.values():
                    if mon.species == switch.species:
                        pos_to_mon[switch_pos] = mon
                        found = True
                        break
                if not found:
                    # Pareto knows the pokemon exists, but it did not enter the field yet
                    switch._set_hp_status("100/100")
                    pos_to_mon[switch_pos] = switch

    return request, pos_to_mon


def get_turn_order(c, pm: PokemonMapper, player) -> List[int]:
    r"""
    Returns a possible turn order (prediction) based on the current moves
    to be perfomed in the genotype for the current turn and the estimated
    speed.
    Args:
        - c: a candidate (genotype) encoding moves and target for each
        attacker
        - pm [PokemonMapper]
        - last_turn: the last turn which has been performed
    Returns:
        turn_order [List[int]]: a list encoding the possible turn order
        containing the position of the attacker that will act (from first
        to last)
    """
    turn_order: List[Tuple[int, int, int]] = []

    # build the turn orders according to the priorities.
    for i in range(0, len(c), 2):
        pos = pm.get_field_pos_from_genotype(i)
        mon = pm.pos_to_mon[pos]
        mon_speed = player.get_mon_estimates(mon, pos)["spe"]
        if isinstance(c[i], Move):
            move = c[i]
            priority = move.priority
        else:
            priority = 6
        turn_order.append((pos, priority, mon_speed))

    # sort the moves based first on their priority and then the speed of the pokemons
    turn_order.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in turn_order]


def map_abstract_target(
    abs_target: int, attacker_pos: int, pm: PokemonMapper
) -> List[int]:
    if abs_target <= 2:
        return [abs_target]
    elif abs_target == 3:
        return [t for t in pm.moves_targets.keys() if (t != attacker_pos)]
    elif abs_target == 4:
        return [t for t in pm.moves_targets.keys() if (t * attacker_pos < 0)]
    elif abs_target == 5:
        return [t for t in pm.moves_targets.keys() if (t * attacker_pos > 0)]
    elif abs_target == 6:
        # actually, we should consider the last pokemon that hit the attacker, but
        # I would say it's an overkill ATM
        candidates = [t for t in pm.moves_targets.keys() if (t * attacker_pos < 0)]
        return sample(candidates, 1)
    else:
        print(
            f"Case {abs_target} not covered, choosing a random target for the computation"
        )
        return sample(pm.moves_targets.keys(), 1)
