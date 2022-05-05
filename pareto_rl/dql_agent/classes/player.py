import numpy as np
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from gym.spaces import Space
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder
from poke_env.environment.battle import Battle

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
    remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / battle.team_size
    remaining_mon_opponent = (
      len([mon for mon in battle.opponent_team.values() if mon.fainted]) / len(battle._opponent_team)
    )
    obs = np.concatenate(
      [moves_base_power, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent]]
    # Final vector with 26 components
    )

    return obs


  def describe_embedding(self) -> Space:
    return super().describe_embedding()

  def calc_reward(self,last_battle, current_battle) -> float:
    return self.reward_computing_helper(last_battle,fainted_value=2,hp_value=1,victory_value=30)-self.reward_computing_helper(current_battle,fainted_value=2,hp_value=1,victory_value=30)

  def get_pokemon_order(self,action,pokemon,battle: Battle) -> BattleOrder:
    # [0,12) max values
    print(f"Action = {action}")
    print(battle.all_active_pokemons)
    if(action < 8 and not battle.force_switch[pokemon]):
      #moves
      print(battle.available_moves)
      move = int(action / 2)
      print(pokemon,move)
      target = DoubleBattle.OPPONENT_1_POSITION if action % 2 == 0 else DoubleBattle.OPPONENT_2_POSITION
      print(self.agent.create_order(battle.available_moves[pokemon][move],move_target=target))
      return self.agent.create_order(battle.available_moves[pokemon][move],move_target=target)
    elif(action >= 8 and not battle.force_switch[pokemon]):
      action = action - 8
      return self.agent.create_order(battle.available_switches[pokemon][action])
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
    print("SONQUASONQUASONQUASONQUASONQUASENDITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTLESSGOOOOOOOOOOOOOOOOOYOLOOOOOOOOOO")
    actions = decode_actions(actions)
    print(battle.force_switch)
    if(actions[0] == -1 or actions[1] == -1):
        return ForfeitBattleOrder()
    first_order = self.get_pokemon_order(actions[0],0,battle)
    second_order = self.get_pokemon_order(actions[1],1,battle)

    return DoubleBattleOrder(first_order,second_order)


def decode_actions(coded_action):
  action1 = coded_action//100
  action2 = coded_action % 100
  return [action1,action2]
