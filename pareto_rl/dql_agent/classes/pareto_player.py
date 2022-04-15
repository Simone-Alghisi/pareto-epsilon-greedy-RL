from poke_env.player.player import Player
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.battle_order import (
    BattleOrder,
)
from poke_env.teambuilder.teambuilder import Teambuilder
from pareto_rl.pareto_front.pareto_search import pareto_search
from argparse import Namespace
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper

class ParetoPlayer(Player):
    r"""ParetoPlayer, should perform only Pareto optimal moves"""

    def __init__(self, **kwargs):
        super(ParetoPlayer, self).__init__(
            battle_format="gen82v2doubles", team=StaticTeambuilder(), **kwargs
        )

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        pokemon_mapper = PokemonMapper(battle)
        args = Namespace(dry=True)
        order = pareto_search(args, battle, pokemon_mapper)
        return self.choose_random_doubles_move(battle)


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
