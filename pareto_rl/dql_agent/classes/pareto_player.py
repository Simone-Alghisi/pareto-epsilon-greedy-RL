from typing import overload
import numpy as np
from poke_env.player.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)
from poke_env.teambuilder.teambuilder import Teambuilder
from pareto_rl.pareto_front.pareto_search import pareto_search
import random


class ParetoPlayer(Player):
    r"""ParetoPlayer, should perform only Pareto optimal moves"""

    def __init__(self, **kwargs):
        super(ParetoPlayer, self).__init__(
            battle_format="gen82v2doubles", team=StaticTeambuilder(), **kwargs
        )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        active_orders = [[], []] # active orders
        for (
            idx,
            (orders, mon, switches, moves, can_mega, can_z_move, can_dynamax),
        ) in enumerate(
            zip(
                active_orders,
                battle.active_pokemon,
                battle.available_switches,
                battle.available_moves,
                battle.can_mega_evolve,
                battle.can_z_move,
                battle.can_dynamax,
            )
        ):
            # if there are pokémons
            if mon:
                # gets all the possible targets
                targets = {
                    move: battle.get_possible_showdown_targets(move, mon)
                    for move in moves
                }
                # merges orders in the following way
                orders.extend(
                    # an array of Battle orders, with move and target
                    # which basically is only a union of move and target pokémon: Optional[Union[Move, Pokemon]]
                    [
                        BattleOrder(move, move_target=target)
                        for move in moves
                        for target in targets[move]
                    ]
                )

                # for ord in orders:
                #    print(f"{ord[0]}{ord[1]}")

                # includes also possible switches
                # orders.extend([(mon, BattleOrder(switch)) for switch in switches])

                # if the poémon can Megaevolve, then we can add this possibility
                # if can_mega:
                #    orders.extend(
                #        [
                #            (mon, BattleOrder(move, move_target=target, mega=True))
                #            for move in moves
                #            for target in targets[move]
                #        ]
                #    )
                
                # the same for the z move
                # if can_z_move:
                #    available_z_moves = set(mon.available_z_moves)
                #    orders.extend(
                #        [
                #            (mon, BattleOrder(move, move_target=target, z_move=True))
                #            for move in moves
                #            for target in targets[move]
                #            if move in available_z_moves
                #        ]
                #    )

                # and obv for dynamax
                # if can_dynamax:
                #    orders.extend(
                #        [
                #            (mon, BattleOrder(move, move_target=target, dynamax=True))
                #            for move in moves
                #            for target in targets[move]
                #        ]
                #    )

                # if you have to switch
                if sum(battle.force_switch) == 1:
                    # A boolean indicating whether the active pokemon is forced to switch out.
                    if orders:
                        # returns a random order
                        return orders[int(random.random() * len(orders))]
                    # basically returns showdown's default move order.
                    # This order will result in the first legal order - according to showdown's
                    # ordering - being chosen.
                    return self.choose_default_move()

        # get the two active pokémons
        first_mon, second_mon = battle.active_pokemon

        # create a double battle order
        orders = DoubleBattleOrder.join_orders(*active_orders)

        # define the Pareto Battle oder, which is basically a touple of 
        # double battle order and respectively the pokémon which performs
        # the chosen moves
        pareto_orders = []
        for ord in orders:
            pareto_orders.append((ord, (first_mon, second_mon)))
        
        if orders:
            # you have to turn the pareto optimal move for that pokémon
            args = lambda: None
            args.dry = True
            return pareto_search(args, orders = pareto_orders)
        else:
            return DefaultBattleOrder()


    # implement abstract method
    # def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # return self.choose_random_doubles_move(battle)


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
