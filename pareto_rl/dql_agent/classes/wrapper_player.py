from typing import Union
from poke_env.player.battle_order import (
    DoubleBattleOrder,
    DefaultBattleOrder,
)
from poke_env.player.player import Player
from pareto_rl.dql_agent.classes.pareto_player import StaticTeambuilder
from pareto_rl.dql_agent.classes.player import CombineActionRLPlayer
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from pareto_rl.dql_agent.utils.teams import VGC_1_3VS3 as TEAM


class WrapperPlayer(Player):
    def __init__(self, model: CombineActionRLPlayer, **kwargs):
        super(WrapperPlayer, self).__init__(
            team=(
                StaticTeambuilder(TEAM)
                if kwargs["battle_format"] == "gen8doublesubers"
                else None
            ),
            **kwargs
        )
        self.model: CombineActionRLPlayer = model
        self.model.policy_net.eval()

    def choose_move(self, battle) -> Union[DoubleBattleOrder, DefaultBattleOrder]:
        state = self.model.embed_battle(battle)
        self.model.pm = PokemonMapper(battle)
        return self.model.decode_action(self.model.policy(state, eps_greedy=False))
