import random
import json
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, DefaultBattleOrder
from poke_env.player.baselines import MaxBasePowerPlayer
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.classes.pareto_player import StaticTeambuilder
from poke_env.environment.move import Move as OriginalMove
from poke_env.environment.pokemon import Pokemon
from pareto_rl.damage_calculator.requester import damage_request_server
from pareto_rl.dql_agent.utils.utils import prepare_pokemon_request
from poke_env.data import MOVES

class DoubleMaxDamagePlayer(MaxBasePowerPlayer):

  def __init__(self, **kwargs):
    super(DoubleMaxDamagePlayer, self).__init__(
      team=(StaticTeambuilder(TEAM) if kwargs['battle_format'] == 'gen8doublesubers' else None),
      **kwargs
    )

  def choose_max_doubles_move(self, battle: DoubleBattle) -> BattleOrder:
    active_orders = [[], []]

    for (
      idx,
      (orders, mon, switches, moves),
    ) in enumerate(
      zip(
        active_orders,
        battle.active_pokemon,
        battle.available_switches,
        battle.available_moves,
      )
    ):
      if mon:
        targets = {
          move: battle.get_possible_showdown_targets(move, mon)
          for move in moves
        }
        orders.extend(
          [
            BattleOrder(move, move_target=target)
            for move in moves
            for target in targets[move]
            # either an attack towards an enemy or a non-attack towards allys
            if target >= 0 or (1 not in targets[move] and 2 not in targets[move])
          ]
        )
        orders.extend([BattleOrder(switch) for switch in switches])

        if sum(battle.force_switch) == 1:
          if orders:
            return orders[int(random.random() * len(orders))]
          return self.choose_default_move()

    orders = DoubleBattleOrder.join_orders(*active_orders)

    data = {}
    data['requests'] = []
    valid_orders = []
    if orders:
      for order in orders:
        if isinstance(order, DefaultBattleOrder):
          return DefaultBattleOrder()
        first_order_request = second_order_request = None
        request = {}
        if order.first_order is not None:
          attacker = battle.active_pokemon[0] if battle.active_pokemon[0] is not None else battle.active_pokemon[1]
          first_order_request = self.get_request(1,order.first_order, attacker, battle)
        if order.second_order is not None:
          attacker = battle.active_pokemon[1]
          second_order_request = self.get_request(2,order.second_order, attacker, battle)
        if first_order_request is not None:
          request.update(first_order_request)
        if second_order_request is not None:
          request.update(second_order_request)
        if request:
          data['requests'].append(request)
          valid_orders.append(order)

      r = damage_request_server(data)
      r = json.loads(r)
      best_dmg = -1
      best_order = None
      for i, request in enumerate(r):
        dmg = 0
        for pos, response in request.items():
          damage = response['damage']
          if isinstance(damage, list):
            damage = damage[len(damage) // 2]
          if pos == '1':
            damage = damage * valid_orders[i].first_order.order.accuracy
          else:
            damage = damage * valid_orders[i].second_order.order.accuracy
          dmg += damage
        if dmg > best_dmg:
          best_dmg = dmg
          best_order = valid_orders[i]
      if best_order is None:
        return DefaultBattleOrder()
      return best_order
    else:
      return DefaultBattleOrder()

  def choose_move(self, battle) -> BattleOrder:
    return self.choose_max_doubles_move(battle)

  def get_request(self, pos: int, order: BattleOrder, attacker: Pokemon, battle: DoubleBattle):
    request = None
    if isinstance(order.order, OriginalMove):
      target_idx = order.move_target - 1
      if target_idx >= 0:
        target = battle.opponent_active_pokemon[target_idx]
        if target is None:
          target = battle.opponent_active_pokemon[(target_idx+1) % 2]
        attacker_args = prepare_pokemon_request(attacker)
        target_args = prepare_pokemon_request(target)
        move_name = MOVES[order.order._id]["name"]
        request = {pos: {
          "attacker": attacker_args,
          "target": target_args,
          "move": move_name,
          "field": {"gameType": "Doubles"},
        }}

    return request

TEAM = """
Calyrex-Shadow @ Focus Sash
Ability: As One (Spectrier)
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Astral Barrage
- Protect
- Will-O-Wisp
- Expanding Force

Indeedee-F (F) @ Psychic Seed
Ability: Psychic Surge
EVs: 248 HP / 8 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Follow Me
- Helping Hand
- Expanding Force
- Protect

Kyogre @ Mystic Water
Ability: Drizzle
EVs: 188 HP / 252 SpA / 4 SpD / 64 Spe
Timid Nature
IVs: 0 Atk
- Protect
- Water Spout
- Origin Pulse
- Ice Beam

Incineroar @ Sitrus Berry
Ability: Intimidate
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Fake Out
- Flare Blitz
- Parting Shot
- Throat Chop

Thundurus (M) @ Assault Vest
Ability: Prankster
EVs: 112 HP / 140 Atk / 4 SpD / 252 Spe
Jolly Nature
- Wild Charge
- Brick Break
- Iron Tail
- Superpower

Urshifu @ Choice Band
Ability: Unseen Fist
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Close Combat
- Sucker Punch
- Wicked Blow
- Poison Jab
"""

TEAM = """
Kyogre @ Mystic Water
Ability: Drizzle
EVs: 188 HP / 252 SpA / 4 SpD / 64 Spe
Timid Nature
IVs: 0 Atk
- Protect
- Water Spout
- Origin Pulse
- Ice Beam

Thundurus (M) @ Assault Vest
Ability: Prankster
EVs: 112 HP / 140 Atk / 4 SpD / 252 Spe
Jolly Nature
- Wild Charge
- Brick Break
- Iron Tail
- Superpower

Urshifu @ Choice Band
Ability: Unseen Fist
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Close Combat
- Sucker Punch
- Wicked Blow
- Poison Jab
"""

TEAM = """
Kyogre @ Mystic Water
Ability: Drizzle
EVs: 188 HP / 252 SpA / 4 SpD / 64 Spe
Timid Nature
IVs: 0 Atk
- Protect
- Water Spout
- Origin Pulse
- Ice Beam

Thundurus (M) @ Assault Vest
Ability: Prankster
EVs: 112 HP / 140 Atk / 4 SpD / 252 Spe
Jolly Nature
- Wild Charge
- Brick Break
- Iron Tail
- Superpower
"""
