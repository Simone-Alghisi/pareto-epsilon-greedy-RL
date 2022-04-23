from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.utils.move import Move
from poke_env.environment.pokemon import Pokemon
from pareto_rl.dql_agent.utils.utils import get_possible_showdown_targets
from typing import Dict, List, Set, Union


class PokemonMapper:
    r"""
    PokemonMapper class which is used in order to retrieve
    basic information which are going to be passed to the pareto
    and could be useful for other applications, e.g.
    - moves_targets: dictionary that given the mon position (int), returns
    the possible targets' position (int) for each of the mon moves
    - mon_to_pos: a dictionary that given a mon returns its field position
    - pos_to_mon: a dictionary that given a position, returns the mon in it
    - mon_indexes: a list used to link mon to genotype
    """

    def __init__(self, battle: DoubleBattle) -> None:
        self.battle = battle
        self.moves_targets: Dict[int, Dict[Move, List[int]]] = {}
        self.mon_to_pos: Dict[Pokemon, int] = {}
        self.pos_to_mon: Dict[int, Pokemon] = {}
        self.mon_indexes: List[int] = []

        # your mons
        pos = -1
        for mon, moves in zip(battle.active_pokemon, battle.available_moves):
            if mon:
                casted_moves: Set[Move] = {Move(move._id) for move in moves}
                # map pokemons with their position
                self.mapper(casted_moves, mon, pos)
            pos -= 1

        # opponent mons
        pos = 1
        for mon in battle.opponent_active_pokemon:
            if mon:
                # hardcoded opponent moves
                moves: Set[Move]
                if mon.species.strip().lower() == "Zigzagoon".strip().lower():
                    moves = {
                        Move("doubleedge"),
                        Move("surf"),
                        Move("bodyslam"),
                        Move("thunderbolt"),
                    }
                else:
                    moves = {
                        Move("energyball"),
                        Move("gigadrain"),
                        Move("knockoff"),
                        Move("leafstorm"),
                    }
                # TODO in the general case, first retrieve known moves and then infer
                # the other probabilistically
                # moves = mon.moves # pokemon used moves
                self.mapper(moves, mon, pos)
            pos += 1

        # remove invalid targets for basic
        for pos, moves in self.moves_targets.items():
            for m, targets in moves.items():
                tmp_targets = targets.copy()
                for t in targets:
                    if (t not in self.moves_targets) and (t < 3):
                        tmp_targets.remove(t)
                self.moves_targets[pos][m] = tmp_targets

    def mapper(self, moves: Set[Move], mon: Pokemon, pos: int) -> None:
        r"""
        Given a mon, its set of moves, and its position of the field, makes
        available additional information to the mapper, i.e.
        - moves_targets: dictionary that given the mon position (int), returns
        the possible targets' position (int) for each of the mon moves
        - mon_to_pos: a dictionary that given a mon returns its field position
        - pos_to_mon: a dictionary that given a position, returns the mon in it
        - mon_indexes: a list used to link mon to genotype
        Args:
            - moves: Set[Move], set containing the moves of a mon
            - mon: Pokemon, the mon that we are considering
            - pos: the position on the field of the mon
        """
        # get moves target
        targets: Dict[Move, List[int]] = {
            move: get_possible_showdown_targets(self.battle, mon, move, pos)
            for move in moves
        }
        # map the target with its position
        self.moves_targets[pos] = targets
        self.mon_to_pos[mon] = pos
        self.pos_to_mon[pos] = mon
        self.mon_indexes.append(pos)

    def alive_pokemon_number(self) -> int:
        r"""
        basic function to return the number of currently alive mons
        on the fied.
        Returns:
            n_alive_mon: the number of mons alive
        """
        return len(self.moves_targets)

    def get_field_pos_from_genotype(self, index: int) -> int:
        r"""
        retrieves the position of the mon associated to a certain
        gene index in the genotype.
        Args:
            - index: the index of a gene in the genotype
        Returns:
            pos: the position of the mon associated to index
        """
        return self.mon_indexes[index // 2]
