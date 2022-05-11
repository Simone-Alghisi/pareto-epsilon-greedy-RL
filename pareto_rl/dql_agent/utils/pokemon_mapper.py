from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
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
        self.original_moves_targets: Dict[int, Dict[Move, List[int]]] = {}
        self.mon_to_pos: Dict[Pokemon, int] = {}
        self.pos_to_mon: Dict[int, Pokemon] = {}
        self.mon_indexes: List[int] = []
        self.available_switches: Dict[int, list] = {}
        self.available_orders: list[DoubleBattleOrder]
        active_orders: List[List[BattleOrder]] = [[], []]

        # your mons
        pos = -1
        for mon, moves, switches, orders in zip(
            battle.active_pokemon,
            battle.available_moves,
            battle.available_switches,
            active_orders,
        ):
            if mon:
                casted_moves: Set[Move] = {Move(move._id) for move in moves}
                # map pokemons with their position
                self.mapper(casted_moves, mon, pos, switches, orders)
            pos -= 1

        # opponent mons
        pos = 1
        for mon in battle.opponent_active_pokemon:
            if mon:
                # hardcoded opponent moves
                opp_moves: Set[Move]
                if mon.species.strip().lower() == "Zigzagoon".strip().lower():
                    opp_moves = {
                        Move("doubleedge"),
                        Move("surf"),
                        Move("bodyslam"),
                        Move("thunderbolt"),
                    }
                else:
                    opp_moves = {
                        Move("energyball"),
                        Move("gigadrain"),
                        Move("knockoff"),
                        Move("leafstorm"),
                    }
                # TODO in the general case, first retrieve known moves and then infer
                # the other probabilistically
                # moves = mon.moves # pokemon used moves
                self.mapper(opp_moves, mon, pos)
            pos += 1

        # remove invalid targets for basic
        for pos, moves in self.moves_targets.items():
            for m, targets in moves.items():
                tmp_targets = targets.copy()
                for t in targets:
                    if (t not in self.moves_targets) and (t < 3):
                        tmp_targets.remove(t)
                self.moves_targets[pos][m] = tmp_targets

        self.available_orders = DoubleBattleOrder.join_orders(*active_orders)

    def mapper(
        self,
        moves: Set[Move],
        mon: Pokemon,
        pos: int,
        available_switches: Union[List[Pokemon], None] = None,
        orders: Union[List[BattleOrder], None] = None,
    ) -> None:
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
        targets: Dict[Move, List[int]] = {}
        original_targets: Dict[Move, List[int]] = {}
        for move in moves:
            ot, pt = get_possible_showdown_targets(self.battle, mon, move, pos)
            targets[move] = pt
            original_targets[move] = ot
            if orders is not None:
                orders.extend(
                    [
                        BattleOrder(move, move_target=target)
                        for target in original_targets[move]
                    ]
                )

        if orders is not None and available_switches is not None:
            orders.extend([BattleOrder(switch) for switch in available_switches])

        # map the target with its position
        self.moves_targets[pos] = targets
        self.original_moves_targets[pos] = original_targets
        self.mon_to_pos[mon] = pos
        self.pos_to_mon[pos] = mon
        self.mon_indexes.append(pos)

        if available_switches is not None:
            self.available_switches[pos] = available_switches

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

    def mask_unavailable_moves(self):
        # TODO, remap moves
        ...
