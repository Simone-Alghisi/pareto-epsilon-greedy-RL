import numpy as np
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from gym.spaces import Space
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder
from poke_env.environment.battle import Battle
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper

class SimpleRLPlayer(Gen8EnvSinglePlayer):
  def __init__(self, **kwargs):
    super(SimpleRLPlayer, self).__init__(**kwargs)

  def embed_battle(self, battle):
    # -1 indicates that the move does not have a base power
    # or is not available
    moves_base_power = -np.ones(8)
    moves_dmg_multiplier = np.ones(16)

    for i,pokemon in enumerate(battle.available_moves):
      pokemon_shift = i*4*2
      for j,move in enumerate(pokemon):
        move_shift = j*2
        moves_base_power[(i*4)+j] = move.base_power / 100 # Simple rescaling to facilitate learning
        if move.type:
          #opponent first pokemon or opponent second pokemon can be None (fainted)
          if(battle.opponent_active_pokemon[0] != None):
            # +0 lmao
            moves_dmg_multiplier[pokemon_shift+move_shift+0] = move.type.damage_multiplier(
              battle.opponent_active_pokemon[0].type_1,
              battle.opponent_active_pokemon[0].type_2,
            )
          if(battle.opponent_active_pokemon[1] != None):
            moves_dmg_multiplier[pokemon_shift+move_shift+1] = move.type.damage_multiplier(
              battle.opponent_active_pokemon[1].type_1,
              battle.opponent_active_pokemon[1].type_2,
            )
    # We count how many pokemons have not fainted in each team
    remaining_mon_team = len([mon for mon in battle.team.values() if not mon.fainted]) / battle.team_size
    remaining_mon_opponent = (
      len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / len(battle._opponent_team)
    )
    obs = np.concatenate(
      [moves_base_power, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent]]
    # Final vector with 26 components
    )

    return obs


  def describe_embedding(self) -> Space:
    return super().describe_embedding()

  def calc_reward(self, last_battle, current_battle) -> float:
    return self.reward_computing_helper(last_battle,fainted_value=2,hp_value=1,victory_value=30)-self.reward_computing_helper(current_battle,fainted_value=2,hp_value=1,victory_value=30)

  def get_pokemon_order(self, action, idx, battle: Battle) -> BattleOrder:
    poke_mapper = PokemonMapper(battle)
    idx_to_pos = {0:-1, 1:-2}
    pos = idx_to_pos[idx]
    n_targets = 5
    n_switches = 4
    mon_actions = 4*n_targets + n_switches
    if pos in poke_mapper.moves_targets:
      moves = [ move for move in poke_mapper.moves_targets[pos].keys() ]
    if(action < (mon_actions - n_switches) and not battle.force_switch[idx]):
      #moves
      move = action // n_targets
      # target = DoubleBattle.OPPONENT_1_POSITION if action % 2 == 0 else DoubleBattle.OPPONENT_2_POSITION
      target = (action % n_targets) - 2
      if move >= len(moves):
        import pdb; pdb.set_trace()
      return self.agent.create_order(moves[move],move_target=target)
    elif(action >= (mon_actions - n_switches) and not battle.force_switch[idx]):
      switch = action - (mon_actions - n_switches)
      # print(idx, switch, battle.available_switches)
      return self.agent.create_order(battle.available_switches[idx][switch])
    else:
      return self.agent.choose_random_move(battle)




  def action_to_move(self, actions, battle: Battle) -> BattleOrder:  # pyre-ignore
    """Converts actions to move orders.
    :param action: The action to convert.
    :type action: int
    :param battle: The battle in which to act.
    :type battle: Battle
    :return: the order to send to the server.
    :rtype: str
    """
    actions = decode_actions(actions)
    if(actions[0] == -1 or actions[1] == -1):
        return ForfeitBattleOrder()
    battle_order = None
    if battle.force_switch[0] or battle.force_switch[1]:
      battle_order = DoubleBattleOrder(self.agent.choose_random_move(battle),None)
    else:
      first_order = None
      second_order = None
      for i, mon in enumerate(battle.active_pokemon):
        if mon:
          if first_order is None:
            first_order = self.get_pokemon_order(actions[i],i,battle)
          else:
            second_order = self.get_pokemon_order(actions[i],i,battle)
      battle_order = DoubleBattleOrder(first_order, second_order)

    # print(battle_order)
    return battle_order


def decode_actions(coded_action):
  action1 = coded_action//100
  action2 = coded_action % 100
  return [action1,action2]
