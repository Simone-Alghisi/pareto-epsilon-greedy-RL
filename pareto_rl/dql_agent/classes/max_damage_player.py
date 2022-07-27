import json
from pareto_rl.damage_calculator.requester import damage_request_server
from pareto_rl.dql_agent.classes.pareto_player import StaticTeambuilder
from pareto_rl.dql_agent.utils.move import Move
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from pareto_rl.dql_agent.utils.teams import VGC_3VS3 as TEAM
from pareto_rl.dql_agent.utils.utils import prepare_pokemon_request
from pareto_rl.pareto_front.classes.next_turn import map_abstract_target
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move as OriginalMove
from poke_env.environment.pokemon import Pokemon
from poke_env.player.baselines import MaxBasePowerPlayer
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)
from typing import List, Optional, Union


class DoubleMaxDamagePlayer(MaxBasePowerPlayer):
    def __init__(self, **kwargs):
        super(DoubleMaxDamagePlayer, self).__init__(
            team=(
                StaticTeambuilder(TEAM)
                if kwargs["battle_format"] == "gen8doublesubers"
                else None
            ),
            **kwargs
        )

    def choose_max_doubles_move(
        self, battle: DoubleBattle
    ) -> Union[DoubleBattleOrder, DefaultBattleOrder]:
        pm: PokemonMapper = PokemonMapper(battle)
        orders = pm.available_orders
        data = {"requests": []}
        valid_orders: List[DoubleBattleOrder] = []
        if orders:
            for order in orders:
                if isinstance(order, DefaultBattleOrder):
                    return DefaultBattleOrder()
                first_order_request = second_order_request = None
                request = {}
                if order.first_order is not None:
                    attacker: Optional[Pokemon] = battle.active_pokemon[0]
                    pos: int = -1
                    if attacker is None:
                        # The mon in pos -1 is dead, long live mon in -2
                        attacker = battle.active_pokemon[1]
                        pos = -2
                    first_order_request = self.get_request(
                        pos, order.first_order, attacker, battle, pm
                    )
                if order.second_order is not None:
                    attacker: Optional[Pokemon] = battle.active_pokemon[1]
                    pos: int = -2
                    second_order_request = self.get_request(
                        -2, order.second_order, attacker, battle, pm
                    )

                if first_order_request is not None:
                    request.update(first_order_request)
                if second_order_request is not None:
                    request.update(second_order_request)

                if request:
                    data["requests"].append(request)
                    valid_orders.append(order)

            r = damage_request_server(data)
            r = json.loads(r)
            best_dmg = -1
            best_order: Union[
                DoubleBattleOrder, DefaultBattleOrder
            ] = DefaultBattleOrder()
            for i, attacks in enumerate(r):
                dmg: float = 0.0
                for idx, response in attacks.items():
                    damage = response["damage"]
                    if isinstance(damage, list):
                        damage = damage[len(damage) // 2]
                    if int(idx) > 0 or valid_orders[i].second_order is None:
                        damage = damage * valid_orders[i].first_order.order.accuracy
                    else:
                        damage = damage * valid_orders[i].second_order.order.accuracy
                    dmg += damage
                if dmg > best_dmg:
                    best_dmg = dmg
                    best_order = valid_orders[i]
            return best_order
        else:
            return DefaultBattleOrder()

    def choose_move(self, battle) -> Union[DoubleBattleOrder, DefaultBattleOrder]:
        return self.choose_max_doubles_move(battle)

    def get_request(
        self,
        pos: int,
        order: BattleOrder,
        attacker: Pokemon,
        battle: DoubleBattle,
        pm: PokemonMapper,
    ) -> dict:
        request = {}
        # consider all attacks that do damage to opponents
        if isinstance(order.order, OriginalMove) and order.move_target >= 0:
            i = 1
            pos_to_idx = {1: 0, 2: 1}
            casted_move = Move(order.order._id)
            if order.move_target == 0:
                # to understand who we are attacking, convert it
                abstract_targets: List[int] = pm.moves_targets[pos][casted_move]
            else:
                abstract_targets = [order.move_target]
            for abstract_target in abstract_targets:
                targets_pos = map_abstract_target(abstract_target, pos, pm)
                for target_pos in targets_pos:
                    # consider only damage to opponents
                    if target_pos in pos_to_idx:
                        target_idx = pos_to_idx[target_pos]
                        target = battle.opponent_active_pokemon[target_idx]
                        if target is None:
                            target = battle.opponent_active_pokemon[
                                (target_idx + 1) % 2
                            ]
                        attacker_args = prepare_pokemon_request(attacker, pos)
                        target_args = prepare_pokemon_request(target, target_pos)
                        move_name = casted_move.get_showdown_name()
                        key = i if pos == -1 else -i
                        request.update(
                            {
                                key: {
                                    "attacker": attacker_args,
                                    "target": target_args,
                                    "move": move_name,
                                    "field": {"gameType": "Doubles"},
                                }
                            }
                        )

        return request
