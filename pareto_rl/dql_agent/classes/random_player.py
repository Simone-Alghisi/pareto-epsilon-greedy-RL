import random
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, DefaultBattleOrder
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.classes.pareto_player import StaticTeambuilder

class DoubleRandomPlayer(RandomPlayer):

  def __init__(self, **kwargs):
    super(DoubleRandomPlayer, self).__init__(
      team=(StaticTeambuilder(TEAM) if kwargs['battle_format'] == 'gen8doublesubers' else None),
      **kwargs
    )

  def choose_random_doubles_move(self, battle: DoubleBattle) -> BattleOrder:
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
          ]
        )
        orders.extend([BattleOrder(switch) for switch in switches])

        if sum(battle.force_switch) == 1:
          if orders:
            return orders[int(random.random() * len(orders))]
          return self.choose_default_move()

    orders = DoubleBattleOrder.join_orders(*active_orders)

    if orders:
      return orders[int(random.random() * len(orders))]
    else:
      return DefaultBattleOrder()

  def choose_move(self, battle) -> BattleOrder:
    return self.choose_random_doubles_move(battle)

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
