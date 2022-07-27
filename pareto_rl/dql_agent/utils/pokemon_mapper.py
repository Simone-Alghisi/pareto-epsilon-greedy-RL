import re
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.battle_order import (
    BattleOrder,
    DoubleBattleOrder,
    DefaultBattleOrder,
)
from pareto_rl.dql_agent.utils.move import Move
from poke_env.environment.move import Move as OriginalMove
from poke_env.environment.pokemon import Pokemon
from poke_env.data import to_id_str
from pareto_rl.dql_agent.utils.utils import (
    get_possible_showdown_targets,
    get_pokemon_showdown_name,
)
from typing import Dict, List, Optional, Union, OrderedDict
from collections import OrderedDict as ordered_dict


class PokemonMapper:
    r"""
    PokemonMapper class which is used in order to retrieve
    basic information which are going to be passed to the pareto
    and could be useful for other applications, e.g.
    - moves_targets: dictionary that given the mon position (int), returns
    the possible targets' position (int) for each of the mon moves
    - pos_to_mon: a dictionary that given a position, returns the mon in it
    - mon_indexes: a list used to link mon to genotype
    """

    def __init__(
        self, battle: DoubleBattle, opponent_team: Optional[str] = None
    ) -> None:
        self.battle = battle
        self.moves_targets: Dict[int, Dict[Move, List[int]]] = {}
        self.original_moves_targets: Dict[int, Dict[Move, List[int]]] = {}
        self.pos_to_mon: OrderedDict[int, Pokemon] = ordered_dict()
        self.mon_indexes: List[int] = []
        self.available_switches: Dict[int, List[Pokemon]] = {}
        self.available_orders: Optional[
            Union[List[DoubleBattleOrder], List[DefaultBattleOrder]]
        ] = None
        self.available_moves: Dict[int, List[Move]] = {}
        self.opponent_info: Optional[Dict[str, List[OriginalMove]]] = self.parse_team(
            opponent_team
        )
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
                # map pokemons with their position
                self.mapper(moves, mon, pos, switches, orders)

                # do not look at me like that, if it breaks it's their fault
                if sum(battle.force_switch) == 1 and self.available_orders is None:
                    if orders:
                        self.available_orders = DoubleBattleOrder.join_orders(
                            orders, None
                        )
                    self.available_orders = [DefaultBattleOrder()]

            pos -= 1

        # Again, if it breaks, not my fault
        if self.available_orders is None:
            self.available_orders = DoubleBattleOrder.join_orders(*active_orders)
            if not self.available_orders:
                self.available_orders = [DefaultBattleOrder()]

        # opponent mons
        pos = 1
        for mon in battle.opponent_active_pokemon:
            if mon:
                if self.opponent_info is not None:
                    opp_moves: List[OriginalMove] = self.opponent_info[
                        get_pokemon_showdown_name(mon)
                    ]
                else:
                    # the other probabilistically
                    opp_moves = list(mon.moves.values())  # pokemon used moves
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

    def mapper(
        self,
        moves: List[OriginalMove],
        mon: Pokemon,
        pos: int,
        available_switches: Optional[List[Pokemon]] = None,
        orders: Optional[List[BattleOrder]] = None,
    ) -> None:
        r"""
        Given a mon, its set of moves, and its position of the field, makes
        available additional information to the mapper, i.e.
        - moves_targets: dictionary that given the mon position (int), returns
        the possible targets' position (int) for each of the mon moves
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
            casted_move: Move = Move(move._id)
            ot, pt = get_possible_showdown_targets(self.battle, mon, casted_move, pos)
            targets[casted_move] = pt
            original_targets[casted_move] = ot
            if pos not in self.available_moves:
                self.available_moves[pos] = []
            self.available_moves[pos].append(casted_move)
            if orders is not None:
                orders.extend(
                    [
                        BattleOrder(move, move_target=target)
                        for target in original_targets[casted_move]
                    ]
                )

        if orders is not None and available_switches is not None:
            orders.extend([BattleOrder(switch) for switch in available_switches])

        # map the target with its position
        self.moves_targets[pos] = targets
        self.original_moves_targets[pos] = original_targets
        self.pos_to_mon[pos] = mon
        self.mon_indexes.append(pos)

        if available_switches is not None:
            self.available_switches[pos] = available_switches
        else:
            if self.opponent_info is not None:
                possible_switches = {mon_name for mon_name in self.opponent_info.keys()}
                for active_opp in self.battle.opponent_active_pokemon:
                    if active_opp:
                        possible_switches.remove(get_pokemon_showdown_name(active_opp))
                for opp in self.battle.opponent_team.values():
                    if opp.fainted:
                        possible_switches.remove(get_pokemon_showdown_name(opp))
                self.available_switches[pos] = [
                    Pokemon(species=to_id_str(showdown_name))
                    for showdown_name in possible_switches
                ]

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

    def parse_team(
        self, opponent_team: Optional[str]
    ) -> Optional[Dict[str, List[OriginalMove]]]:
        if opponent_team is None:
            return None
        opponent_info: Dict[str, List[OriginalMove]] = {}
        coarse_split = re.split(r"Ability|\n\n", opponent_team[1:-1])
        for i in range(0, len(coarse_split), 2):
            mon_name = re.split(r"@|\n|\(| ", coarse_split[i])[0]
            opponent_info[mon_name] = [
                OriginalMove(Move.retrieve_id(re.split(r"\n", move_name)[0]))
                for move_name in re.split(r"- ", coarse_split[i + 1])[-4:]
            ]
        return opponent_info
