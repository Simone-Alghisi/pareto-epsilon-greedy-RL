import csv
import os
import random
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
from typing import Dict, List, Optional, Set, Union, OrderedDict, Tuple
from collections import OrderedDict as ordered_dict
from pokemon_info.scraper import scrape_mon_data


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
        self,
        battle: DoubleBattle,
        showdown_opp_team: Optional[str] = None,
    ) -> None:
        self.battle: DoubleBattle = battle
        self.opponent_info: Optional[Dict[str, List[OriginalMove]]] = self.parse_team(
            showdown_opp_team
        )
        self.moves_targets: Dict[int, Dict[Move, List[int]]] = {}
        self.original_moves_targets: Dict[int, Dict[Move, List[int]]] = {}
        self.pos_to_mon: OrderedDict[int, Pokemon] = ordered_dict()
        self.mon_indexes: List[int] = []
        self.available_switches: Dict[int, List[Pokemon]] = {}
        self.available_orders: Optional[
            Union[List[DoubleBattleOrder], List[DefaultBattleOrder]]
        ] = None
        self.available_moves: Dict[int, List[Move]] = {}
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
                    opp_moves_names = [
                        Move(move._id).get_showdown_name() for move in opp_moves
                    ]
                    missing_moves = 4 - len(opp_moves)
                    if missing_moves > 0:
                        showdown_name = get_pokemon_showdown_name(mon)
                        path = (
                            f"./pokemon-data/{showdown_name}/{showdown_name}_Moves.csv"
                        )
                        possible_moves: OrderedDict[str, float] = ordered_dict()
                        tmp: List[Tuple[str, float]] = []
                        if not os.path.isfile(path):
                          try:
                            scrape_mon_data([showdown_name])
                          except:
                            print(f'Failed to download info for {showdown_name}')
                        if os.path.isfile(path):
                            with open(path) as move_file:
                                reader = csv.DictReader(
                                    move_file, ["name", "type", "usage"]
                                )
                                for line in reader:
                                    if (
                                        line["name"] != "Other"
                                        and line["name"] not in opp_moves_names
                                    ):
                                        usage = float(line["usage"][:-1]) / 100
                                        tmp.append((line["name"], usage))
                            while len(tmp) > 0:
                                move_name, usage = tmp.pop()
                                possible_moves[move_name] = usage
                            # extract moves
                            for _ in range(missing_moves):
                                opp_moves.append(
                                    self.extract_weighted_move(possible_moves)
                                )
                        else:
                            opp_moves.append(OriginalMove("tackle"))
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

        tmp: Dict[int, Dict[Move, List[int]]] = {}
        for pos, moves in self.moves_targets.items():
            for m, targets in moves.items():
                if len(targets) > 0:
                    if pos not in tmp:
                        tmp[pos] = {}
                    tmp[pos][m] = targets

        self.moves_targets = tmp


    def extract_weighted_move(
        self, possible_moves: OrderedDict[str, float]
    ) -> OriginalMove:
        rand_val = random.random()
        total = sum(list(possible_moves.values()))
        summed_prob = 0
        # default to avoid running out of possible_moves
        move_name = "tackle"
        for k, v in possible_moves.items():
            summed_prob += v / total
            if rand_val < summed_prob:
                del possible_moves[k]
                move_name = k
                break
        return OriginalMove(Move.retrieve_id(move_name))

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
        # Damn you, Urshifu-*, Zacian-*, and Zamazenta-*!
        full_team: Set[str] = {get_pokemon_showdown_name(mon).split("-*")[0] for mon in self.battle._teampreview_opponent_team}
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
            possible_switches: List[str] = []
            active_opps: Set[str] = {
                get_pokemon_showdown_name(active_opp)
                for active_opp in self.battle.opponent_active_pokemon
                if active_opp
            }
            known_opp: Set[str] = set()

            for opp in self.battle.opponent_team.values():
                showdown_name: str = get_pokemon_showdown_name(opp)
                got_censored: bool = False
                for censored in ["Zacian", "Zamazenta", "Urshifu"]:
                    if showdown_name.startswith(censored):
                        known_opp.add(showdown_name.split("-")[0])
                        got_censored = True
                        break
                if not got_censored:
                    known_opp.add(showdown_name)
                if not opp.fainted and not showdown_name in active_opps:
                    possible_switches.append(showdown_name)

            remaining_opps: List[str] = list(full_team.difference(known_opp))
            to_sample: int = self.battle.max_team_size - len(known_opp)
            possible_switches.extend(random.sample(remaining_opps, to_sample))

            self.available_switches[pos] = [
                Pokemon(species=to_id_str(showdown_name))
                for showdown_name in possible_switches
            ]

    def alive_pokemon_number(self) -> int:
        r"""
        basic function to return the number of currently alive mons
        on the field.
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

    def get_gene_idx_from_field_pos(self, pos: int) -> int:
        r"""
        retrieves the gene index of the mon associated to a certain
        field position.
        Args:
            - pos: the position of the mon associated to index
        Returns:
            index: the index of a gene in the genotype
        """
        return self.mon_indexes.index(pos) * 2

    def parse_team(
        self, showdown_opp_team: Optional[str]
    ) -> Optional[Dict[str, List[OriginalMove]]]:
        if showdown_opp_team is None:
            return None
        opponent_info: Dict[str, List[OriginalMove]] = {}
        coarse_split = re.split(r"Ability|\n\n", showdown_opp_team[1:-1])
        for i in range(0, len(coarse_split), 2):
            mon_name = re.split(r"@|\n|\(| ", coarse_split[i])[0]
            opponent_info[mon_name] = [
                OriginalMove(Move.retrieve_id(re.split(r"\n", move_name)[0]))
                for move_name in re.split(r"- ", coarse_split[i + 1])[-4:]
            ]
        return opponent_info
