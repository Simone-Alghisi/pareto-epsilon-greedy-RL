from typing import overload
import numpy as np
from poke_env.player.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.teambuilder.teambuilder import Teambuilder
from pareto_rl.pareto_front import pareto_search

class ParetoPlayer(Player):
  r"""ParetoPlayer
  """
  def __init__(self, **kwargs):
    super(ParetoPlayer, self).__init__(battle_format='gen82v2doubles', team=StaticTeambuilder(), **kwargs)

  # implement abstract method
  def choose_move(self, battle: AbstractBattle):
    return self.choose_random_doubles_move(battle)
    # TODO, should do something like the pareto move
    #return pareto_search()

# qui

TEAM = """
Zigzagoon-Galar @ Aguav Berry
Ability: Quick Feet
EVs: 252 HP / 136 Atk / 120 SpA
- Double-Edge
- Surf
- Knock Off
- Thunderbolt

Charmander @ Aguav Berry
Ability: Blaze
EVs: 252 HP / 136 Atk / 120 SpA
- Fire Blast
- Fire Fang
- Fling
- Seismic Toss
"""

class StaticTeambuilder(Teambuilder):
  def yield_team(self):
    team = self.join_team(self.parse_showdown_team(TEAM))
    print(team)
    return team


# https://github.com/hsahovic/poke-env/blob/c5464ab43719af4e24af835c446e15b78c022348/src/poke_env/player/player.py
#def choose_random_doubles_move(self, battle: DoubleBattle) -> BattleOrder:
#    active_orders = [[], []]
#
#    for (
#        idx,
#        (orders, mon, switches, moves, can_mega, can_z_move, can_dynamax),
#    ) in enumerate(
#        zip(
#            active_orders,
#            battle.active_pokemon,
#            battle.available_switches,
#            battle.available_moves,
#            battle.can_mega_evolve,
#            battle.can_z_move,
#            battle.can_dynamax,
#        )
#    ):
#        if mon:
#            targets = {
#                move: battle.get_possible_showdown_targets(move, mon)
#                for move in moves
#            }
#            orders.extend(
#                [
#                    BattleOrder(move, move_target=target)
#                    for move in moves
#                    for target in targets[move]
#                ]
#            )
#            orders.extend([BattleOrder(switch) for switch in switches])
#
#            if can_mega:
#                orders.extend(
#                    [
#                        BattleOrder(move, move_target=target, mega=True)
#                        for move in moves
#                        for target in targets[move]
#                    ]
#                )
#            if can_z_move:
#                available_z_moves = set(mon.available_z_moves)
#                orders.extend(
#                    [
#                        BattleOrder(move, move_target=target, z_move=True)
#                        for move in moves
#                        for target in targets[move]
#                        if move in available_z_moves
#                    ]
#                )
#
#            if can_dynamax:
#                orders.extend(
#                    [
#                        BattleOrder(move, move_target=target, dynamax=True)
#                        for move in moves
#                        for target in targets[move]
#                    ]
#                )
#
#            if sum(battle.force_switch) == 1:
#                if orders:
#                    return orders[int(random.random() * len(orders))]
#                return self.choose_default_move()
#
#    orders = DoubleBattleOrder.join_orders(*active_orders)
#
#    if orders:
#        return orders[int(random.random() * len(orders))]
#    else:
#        return DefaultBattleOrder()