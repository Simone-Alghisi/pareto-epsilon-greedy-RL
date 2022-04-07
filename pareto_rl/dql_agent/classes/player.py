import numpy as np
from poke_env.player.env_player import Gen8EnvSinglePlayer

class SimpleRLPlayer(Gen8EnvSinglePlayer):
  def __init__(self, **kwargs):
    super(SimpleRLPlayer, self).__init__(**kwargs)

  def embed_battle(self, battle):
    # -1 indicates that the move does not have a base power
    # or is not available
    moves_base_power = -np.ones(4)
    moves_dmg_multiplier = np.ones(4)
    for i, move in enumerate(battle.available_moves):
      moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
      if move.type:
        moves_dmg_multiplier[i] = move.type.damage_multiplier(
          battle.opponent_active_pokemon.type_1,
          battle.opponent_active_pokemon.type_2,
        )

    # We count how many pokemons have not fainted in each team
    remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
    remaining_mon_opponent = (
      len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
    )

    obs = np.concatenate(
      [moves_base_power, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent]]
    # Final vector with 10 components
    )

    return obs

  def compute_reward(self, battle) -> float:
    return self.reward_computing_helper(
      battle,
      fainted_value=2,
      hp_value=1,
      victory_value=30,
    )
