import random
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, DefaultBattleOrder
from poke_env.player.baselines import MaxBasePowerPlayer
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.classes.pareto_player import StaticTeambuilder
from poke_env.environment.move import Move as OriginalMove
from poke_env.environment.pokemon import Pokemon

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

    if orders:
      best_dmg = -1
      best_order = None
      for order in orders:
        first_order_dmg = 0
        second_order_dmg = 0
        if order.first_order is not None:
          attacker = battle.active_pokemon[0] if battle.active_pokemon[0] is not None else battle.active_pokemon[1]
          first_order_dmg = self.calc_dmg(order.first_order, attacker, battle)
        if order.second_order is not None:
          attacker = battle.active_pokemon[1]
          second_order_dmg = self.calc_dmg(order.first_order, attacker, battle)
        dmg = first_order_dmg + second_order_dmg
        if dmg > best_dmg:
          best_dmg = dmg
          best_order = order
      return best_order
    else:
      return DefaultBattleOrder()

  def choose_move(self, battle) -> BattleOrder:
    return self.choose_max_doubles_move(battle)

  def calc_dmg(self, order: BattleOrder, attacker: Pokemon, battle: DoubleBattle):
    damage = 0
    if isinstance(order.order, OriginalMove):
      target_idx = order.move_target - 1
      if target_idx > 0:
        damage = order.order.base_power
        if order.order.type in attacker.types:
          damage *= 1.5
        target = battle.opponent_active_pokemon[target_idx]
        if target is None:
          target = battle.opponent_active_pokemon[(target_idx+1) % 2]
        damage *= order.order.type.damage_multiplier(target.type_1, target.type_2)
    return damage

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
"""
