import torch
import random
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from gym.spaces import Space
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder, DoubleBattleOrder, ForfeitBattleOrder
from poke_env.environment.battle import Battle
from pareto_rl.dql_agent.utils.move import Move
from poke_env.environment.pokemon import Pokemon
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from typing import Any, Dict, List, Union
from pareto_rl.dql_agent.classes.darkr_ai import DarkrAI, Transition, ReplayMemory
from abc import ABC, abstractmethod
from poke_env.environment.move import Move as OriginalMove
from pareto_rl.dql_agent.classes.pareto_player import StaticTeambuilder
from pareto_rl.dql_agent.classes.pareto_player import AsyncParetoPlayer

class SimpleRLPlayer(Gen8EnvSinglePlayer):
  def __init__(self, **kwargs):
    super(SimpleRLPlayer, self).__init__(**kwargs)

  def embed_battle(self, battle: DoubleBattle) -> list:
    obs = []
    active = []
    bench = []
    # labels = []

    # obs.append(len([mon for mon in battle.team.values() if not mon.fainted])/len(battle.team))
    # labels.append("remaning_mon")
    # obs.append(len([mon for mon in battle.opponent_team.values() if not mon.fainted])/len(battle.opponent_team))
    # labels.append("remaning_opp")

    for i,mon in enumerate(battle.team.values()):
      # lots of info are available, the problem is time,
      # hughes effect, and also mapping
      mon_data = []
      # bl = f"mon_{i}"

      # types (2)
      # types = [t.value / 18 if t is not None else 0 for t in mon.types]
      # mon_data.extend(types)
      # labels.extend([f"{bl}_type_1", f"{bl}_type_2"])

      # hp normalised (good idea?)
      mon_data.append(mon.current_hp_fraction)
      # labels.append(f"{bl}_hp_frac")

      # stats (5)
      # mon_data.extend([stat / 614 for stat in mon.stats.values()])
      # labels.extend([f"{bl}_{stat}" for stat in mon.stats.keys()])

      # boosts and debuffs (7)
      # TODO it may be possible to compute it together with
      # the stats above to reduce the parameters
      # mon_data.extend([(boost+6)/12 for boost in mon.boosts.values()])
      # labels.extend([f"{bl}_{boost}" for boost in mon.boosts.keys()])

      # for stat_name, stat in mon.stats.items():
      #   boost = mon.boosts[stat_name]
      #
      #   # labels.append(f"{bl}_{stat_name}_w_boost_mult")
      #   if boost >= 0:
      #     mon_data.append(stat/614*((2+boost)/2))
      #   else:
      #     mon_data.append(stat/614*(2/(2+(-boost))))

      # status
      # TODO one-hot-encoding?
      # mon_data.append(mon.status.value / 7 if mon.status is not None else 0)
      # labels.append(f"{bl}_status")

      # moves
      # TODO... is it possible to have less than 4?
      j = 0
      # mbl = f"move_{j}"
      for j, move in enumerate(mon.moves.values()):
        # mbl = f"move_{j}"
        move_data = []

        # TODO... should we consider insering the move id?
        # while it may be difficult to learn...
        # it may be particularly useful to discriminate
        # the final effect of the move if similar
        # N.B this is a string, need to convert it using
        # the MOVES dictionary from the json
        # move_data.append(move._id)

        # base power
        move_damage = move.base_power
        # consider STAB (same type attack bonus)
        if move.type in mon.types:
          move_damage *= 1.5
        # move_data.append(move_damage / 100) # normalisation
        # labels.append(f"{bl}_{mbl}_damage")

        # priority
        # move_data.append((move.priority + 7)/12)
        # labels.append(f"{bl}_{mbl}_priority")

        # accuracy
        # TODO... should we encode together w/ damage?
        # move_data.append(move.accuracy)
        # labels.append(f"{bl}_{mbl}_accuracy")

        # category (?)
        # move_data.append((move.category.value - 1) / 2)
        # labels.append(f"{bl}_{mbl}_category")

        # pp (?)
        # move_data.append(move.current_pp / move.max_pp)
        # labels.append(f"{bl}_{mbl}_remaining_pp_percentage")

        # recoil (?)
        # move_data.append(move.recoil)
        # labels.append(f"{bl}_{mbl}_recoil")

        # damage for each active opponent (2)
        for k, opp in enumerate(battle.opponent_active_pokemon):
          # labels.append(f"{bl}_{mbl}_opp_{k}_dmg")
          if opp is not None:
            mlt = move.type.damage_multiplier(opp.type_1, opp.type_2)
            move_data.append(move_damage*mlt*move.accuracy/100) # normalisation
          else:
            # if one is dead, append -1
            move_data.append(-1)

        mon_data.extend(move_data)

      # mon_data.extend([-1 for _ in range(8)]*(4-len(mon.moves)))
      mon_data.extend([-1 for _ in range(2)]*(4-len(mon.moves)))

      # for l in range(4-len(mon.moves)):
      #   # labels.extend([f"{bl}_move_{l}_damage",
      #     f"{bl}_move_{l}_priority",
      #     f"{bl}_move_{l}_accuracy",
      #     f"{bl}_move_{l}_category",
      #     f"{bl}_move_{l}_remaining_pp_percentage",
      #     f"{bl}_move_{l}_recoil",
      #     f"{bl}_move_{l}_act_1_dmg",
      #     f"{bl}_move_{l}_act_2_dmg"])

      if mon.active == True:
        active.extend(mon_data)
      else:
        bench.extend(mon_data)

    obs.extend(active)
    obs.extend(bench)

    active = []
    bench = []

    for i, mon in enumerate(battle.opponent_team.values()):
      # lots of info are available, the problem is time,
      # hughes effect, and also mapping
      mon_data = []
      # bl = f"opp_{i}"

      # types (2)
      # types = [t.value / 18 if t is not None else 0 for t in mon.types]
      # mon_data.extend(types)
      # labels.extend([f"{bl}_type_1", f"{bl}_type_2"])

      # hp normalised (good idea?)
      mon_data.append(mon.current_hp_fraction)
      # labels.append(f"{bl}_hp_frac")

      # stats (5)
      # mon_data.extend([stat / 230 for stat in mon.base_stats.values()])
      # labels.extend([f"{bl}_{stat}" for stat in mon.base_stats.keys()])

      # boosts and debuffs (7)
      # TODO it may be possible to compute it together with
      # the stats above to reduce the parameters
      # mon_data.extend([(boost+6)/12 for boost in mon.boosts.values()])
      # labels.extend([f"{bl}_{boost}" for boost in mon.boosts.keys()])

      # for stat_name, stat in mon.base_stats.items():
      #   if stat_name == "hp":
      #     continue
      #   boost = mon.boosts[stat_name]
      #
      #   # labels.append(f"{bl}_{stat_name}_w_boost_mult")
      #   if boost >= 0:
      #     mon_data.append(stat/230*((2+boost)/2))
      #   else:
      #     mon_data.append(stat/230*(2/(2+(-boost))))

      # status
      # TODO one-hot-encoding?
      # mon_data.append(mon.status.value / 7 if mon.status is not None else 0)
      # labels.append(f"{bl}_status")

      # moves
      # TODO... is it possible to have less than 4?
      j = 0
      # mbl = f"move_{j}"
      for j, move in enumerate(mon.moves.values()):
        # mbl = f"move_{j}"
        move_data = []

        # TODO... should we consider insering the move id?
        # while it may be difficult to learn...
        # it may be particularly useful to discriminate
        # the final effect of the move if similar
        # N.B this is a string, need to convert it using
        # the MOVES dictionary from the json
        # move_data.append(move._id)

        # base power
        move_damage = move.base_power
        # consider STAB (same type attack bonus)
        if move.type in mon.types:
          move_damage *= 1.5
        # move_data.append(move_damage / 100) # normalisation
        # labels.append(f"{bl}_{mbl}_damage")

        # priority
        # move_data.append((move.priority + 7)/12)
        # labels.append(f"{bl}_{mbl}_priority")

        # accuracy
        # TODO... should we encode together w/ damage?
        # move_data.append(move.accuracy)
        # labels.append(f"{bl}_{mbl}_accuracy")

        # category (?)
        # move_data.append((move.category.value - 1) / 2)
        # labels.append(f"{bl}_{mbl}_category")

        # pp (?)
        # move_data.append(move.current_pp / move.max_pp)
        # labels.append(f"{bl}_{mbl}_remaining_pp_percentage")

        # recoil (?)
        # move_data.append(move.recoil)
        # labels.append(f"{bl}_{mbl}_recoil")

        # damage for each active opponent (2)
        for k, opp in enumerate(battle.active_pokemon):
          # labels.append(f"{bl}_{mbl}_act_{k}_dmg")
          if opp is not None:
            mlt = move.type.damage_multiplier(opp.type_1, opp.type_2)
            move_data.append(move_damage*mlt*move.accuracy/100) # normalisation
          else:
            # if one is dead, append -1
            move_data.append(-1)

        mon_data.extend(move_data)

      mon_data.extend([-1 for _ in range(2)]*(4-len(mon.moves)))
      # for l in range(4-len(mon.moves)):
      #   # labels.extend([f"{bl}_move_{l}_damage",
      #     f"{bl}_move_{l}_priority",
      #     f"{bl}_move_{l}_accuracy",
      #     f"{bl}_move_{l}_category",
      #     f"{bl}_move_{l}_remaining_pp_percentage",
      #     f"{bl}_move_{l}_recoil",
      #     f"{bl}_move_{l}_act_1_dmg",
      #     f"{bl}_move_{l}_act_2_dmg"])

      if mon.active == True:
        active.extend(mon_data)
      else:
        bench.extend(mon_data)

    # 3v3
    # bench.extend([-1 for _ in range(9)]*(1-len(bench)))
    obs.extend(active)
    obs.extend(bench)

    # we could also take into account the opponents
    # (which I would say is mandatory)
    # and the field conditions (at least some of them)
    # return (obs, labels)
    return obs

  def describe_embedding(self) -> Space:
    return super().describe_embedding()

  def calc_reward(self, last_battle, current_battle) -> float:
    return self.reward_computing_helper(current_battle,fainted_value=2,hp_value=1,victory_value=30)
    # return self.reward_computing_helper(current_battle,fainted_value=5,hp_value=1,victory_value=60,status_value=0.5)

class BaseRLPlayer(SimpleRLPlayer, ABC):
  def __init__(
      self,
      n_switches: int,
      n_moves: int,
      n_targets: int,
      eps_start: float,
      eps_end: float,
      eps_decay: float,
      batch_size: int,
      gamma: float,
      **kwargs):
    super(BaseRLPlayer, self).__init__(
      team=(StaticTeambuilder(TEAM) if kwargs['battle_format'] == 'gen8doublesubers' else None),
      **kwargs
    )
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.n_targets = n_targets
    self.n_switches = n_switches
    self.n_moves = n_moves
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.eps_threshold = 1
    self.batch_size = batch_size
    self.gamma = gamma
    self.n_actions = None
    self.output_size = None
    self.pm = None

  def _init_model(self, input_size, hidden_layers):
    self.policy_net = DarkrAI(input_size, hidden_layers, self.output_size).to(self.device)
    self.target_net = DarkrAI(input_size, hidden_layers, self.output_size).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimiser = optim.Adam(self.policy_net.parameters())

  def update_pm(self):
    self.pm = PokemonMapper(self.current_battle)

  def update_target(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

  @abstractmethod
  def optimize_model(self, memory: ReplayMemory):
    pass

  @abstractmethod
  def policy(self, state, step: int = 0, eps_greedy: bool = True):
    pass



class DoubleActionRLPlayer(BaseRLPlayer):
  def __init__(
      self,
      input_size: int,
      hidden_layers: List[int],
      n_switches: int,
      n_moves: int,
      n_targets: int,
      eps_start: float,
      eps_end: float,
      eps_decay: float,
      batch_size: int,
      gamma: float,
      **kwargs
      ):
    super(DoubleActionRLPlayer, self).__init__(n_switches,n_moves,n_targets,eps_start,eps_end,eps_decay,batch_size,gamma,**kwargs)
    self.output_size = (n_moves*n_targets + n_switches)*2
    self.n_actions = self.output_size // 2
    self._init_model(input_size,hidden_layers)

  def get_pokemon_order(self, action, idx, battle: DoubleBattle) -> BattleOrder:
    idx_to_pos = {0:-1, 1:-2}
    pos = idx_to_pos[idx]
    if pos in self.pm.moves_targets:
      moves = [ move for move in self.pm.moves_targets[pos].keys() ]
    if(action < (self.n_actions - self.n_switches) and not battle.force_switch[idx]):
      #moves
      move = action // self.n_targets
      # target = DoubleBattle.OPPONENT_1_POSITION if action % 2 == 0 else DoubleBattle.OPPONENT_2_POSITION
      target = (action % self.n_targets) - 2
      if move >= len(moves):
        # import pdb; pdb.set_trace()
        pass
      return self.agent.create_order(moves[move],move_target=target)
    elif(action >= (self.n_actions - self.n_switches) and not battle.force_switch[idx]):
      switch = action - (self.n_actions - self.n_switches)
      # print(idx, switch, battle.available_switches)
      if switch >= len(battle.available_switches[idx]):
        # import pdb; pdb.set_trace()
        pass
      return self.agent.create_order(battle.available_switches[idx][switch])
    else:
      return self.agent.choose_random_move(battle)

  def action_to_move(self, actions, battle: DoubleBattle) -> BattleOrder:  # pyre-ignore
    """Converts actions to move orders.
    :param action: The action to convert.
    :type action: int
    :param battle: The battle in which to act.
    :type battle: Battle
    :return: the order to send to the server.
    :rtype: str
    """
    actions = self._decode_actions(actions)
    if(actions[0] == -1 or actions[1] == -1):
        return ForfeitBattleOrder()
    battle_order = None
    if battle.force_switch[0] or battle.force_switch[1]:
      battle_order = DoubleBattleOrder(self.agent.choose_random_move(battle),None)
    else:
      try:
        first_order = None
        second_order = None
        for i, mon in enumerate(battle.active_pokemon):
          if mon:
            if first_order is None:
              first_order = self.get_pokemon_order(actions[i],i,battle)
            else:
              second_order = self.get_pokemon_order(actions[i],i,battle)
        battle_order = DoubleBattleOrder(first_order, second_order)
      except:
        battle_order = ForfeitBattleOrder()

    # print(battle_order)
    return battle_order

  def optimize_model(self, memory: ReplayMemory):
    if len(memory) < self.batch_size:
      return
    self.policy_net.train()
    self.policy_net.zero_grad()
    self.optimiser.zero_grad()
    transitions = memory.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = [[],[]]
    for action in batch.action:
      action_batch[0].append(action[0])
      action_batch[1].append(action[1])
    action_batch = torch.tensor(action_batch, device=self.device)
    reward_batch = torch.cat(batch.reward).double()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    utility = self.policy_net(state_batch)
    # For each selected action, select its corresponding utility value
    state_action_values = [
        utility.gather(1,action_batch[0].unsqueeze(1)).squeeze(),
        utility.gather(1,action_batch[1].unsqueeze(1)).squeeze()
    ]

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = [
        torch.zeros(self.batch_size, dtype=torch.float64, device=self.device),
        torch.zeros(self.batch_size, dtype=torch.float64, device=self.device)
    ]
    # Select greedily max action (off-policy, Q-Learning)
    next_state_utility = self.target_net(non_final_next_states)
    # Split for first and second player actions
    next_state_utility = torch.split(next_state_utility, split_size_or_sections=next_state_utility.shape[1] // 2, dim=1)
    next_state_values[0][non_final_mask] = next_state_utility[0].max(1)[0]
    next_state_values[1][non_final_mask] = next_state_utility[1].max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = [
        (next_state_values[0] * self.gamma) + reward_batch,
        (next_state_values[1] * self.gamma) + reward_batch
    ]

    # Compute Huber loss
    criterion = nn.HuberLoss()
    loss = criterion(state_action_values[0], expected_state_action_values[0]) \
         + criterion(state_action_values[1], expected_state_action_values[1])
    loss_cpy = loss.detach()

    # Optimize the model
    loss.backward()
    for param in self.policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    self.optimiser.step()

    return loss_cpy

  def _is_valid(self, pos:int, encoded_move_idx: int):
    valid = False
    n_targets = self.n_targets
    if pos in self.pm.original_moves_targets:
      #TODO check if order is the same here and in rlplayer
      moves = [targets for _, targets in self.pm.original_moves_targets[pos].items()]

      if encoded_move_idx >= self.n_actions - self.n_switches:
        # check validity of switch
        # TODO check if benched pokemons' order/position matters
        n_alive = len([mon for mon in self.current_battle.team.values() if not mon.fainted])
        if n_alive > 2:
          max_switch = n_alive - 2
          switch_target = encoded_move_idx - (self.n_actions - self.n_switches)
          if switch_target < max_switch:
            if pos in self.pm.available_switches and len(self.pm.available_switches[pos]) > 0:
              valid = True
      else:
        # check validity of attack
        move_idx = encoded_move_idx // n_targets
        target = (encoded_move_idx % n_targets) - 2
        if move_idx < len(moves):
          if target in moves[move_idx]:
            valid = True
    return valid


  def _get_valid_actions(self, actions: torch.Tensor, utilities: torch.Tensor):
    valid_moves = False
    move = [0,0]
    ally_pos = [pos for pos in self.pm.pos_to_mon.keys() if pos < 0]
    pos_to_idx = {-1: 0, -2: 1}
    # TODO consider dead pokemon
    while not valid_moves:
      # get first valid actions for both players
      for pos in ally_pos:
        # mon = -2 if mon == 0 else -1
        idx = pos_to_idx[pos]
        while (
            move[idx] < self.n_actions and
            not self._is_valid(pos, actions[idx][move[idx]].item())
        ):
          move[idx] += 1

      if move[0] < self.n_actions and move[1] < self.n_actions:
        # check if both moves are switches and if they are the same
        if(actions[0][move[0]] >= (self.n_actions - self.n_switches) and actions[1][move[1]] >= (self.n_actions - self.n_switches)):
          #both switch actions
          if(actions[0][move[0]] == actions[1][move[1]]):
            if(utilities[0][move[0]] > utilities[1][move[1]]):
              #get next best action for second utility
              move[1] += 1
            else:
              move[0] += 1
          else:
            valid_moves = True
        else:
          valid_moves = True
      else:
        # TODO seems that it happenes only with one pokemon available
        break

    if not valid_moves:
      return torch.tensor([0,0])
    else:
      return torch.tensor([ actions[idx][move[idx]] for idx in [0,1] ])

  def policy(self, state, step: int = 0, eps_greedy: bool = True):
    self.policy_net.eval()

    if eps_greedy:
      sample = random.random()
      eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * step / self.eps_decay)
      self.eps_threshold = eps_threshold

    if not eps_greedy or sample > eps_threshold:
    # if True:
      with torch.no_grad():
        #divide output in 2 halves, select the max in the first half and the max in the second half.
        output = self.policy_net(state)
        split_index = len(output) // 2
        # order actions and utilities
        outputs = [output[:split_index].sort(descending=True), output[split_index:].sort(descending=True)]
        utilities = torch.stack([ tensor.values for tensor in outputs ])
        actions = torch.stack([ tensor.indices for tensor in outputs ])
        return self._get_valid_actions(actions,utilities)
    else:
      # TODO maybe more efficient getting random valid action then decoding
      actions = torch.stack([torch.randperm(self.n_actions) for _ in range(2)])
      utilities = torch.rand(2,self.n_actions)
      return self._get_valid_actions(actions,utilities)

  def _encode_actions(self, actions):
    return actions[0]*100 + actions[1]

  def _decode_actions(self, coded_action):
    action1 = coded_action//100
    action2 = coded_action % 100
    return [action1,action2]


class CombineActionRLPlayer(BaseRLPlayer):
  def __init__(
      self,
      input_size: int,
      hidden_layers: List[int],
      n_switches: int,
      n_moves: int,
      n_targets: int,
      eps_start: float,
      eps_end: float,
      eps_decay: float,
      batch_size: int,
      gamma: float,
      **kwargs
      ):
    super(CombineActionRLPlayer, self).__init__(n_switches,n_moves,n_targets,eps_start,eps_end,eps_decay,batch_size,gamma,**kwargs)
    self.n_actions = n_moves * n_targets + n_switches
    self.output_size = (n_moves * n_targets) * self.n_actions + n_switches * (self.n_actions -1) + self.n_actions * 2 + 1
    self._init_model(input_size,hidden_layers)

  def action_to_move(self, action, battle: DoubleBattle) -> BattleOrder:  # pyre-ignore
    return self.decode_action(action)

  def optimize_model(self, memory: ReplayMemory):
    if len(memory) < self.batch_size:
      return
    self.policy_net.train()
    self.policy_net.zero_grad()
    self.optimiser.zero_grad()
    transitions = memory.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.tensor(batch.action, device=self.device)
    reward_batch = torch.cat(batch.reward).double()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    utility = self.policy_net(state_batch)
    # For each selected action, select its corresponding utility value
    state_action_values = utility.gather(1,action_batch.unsqueeze(1)).squeeze()

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(self.batch_size, dtype=torch.float64, device=self.device)
    # Select greedily max action (off-policy, Q-Learning)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.HuberLoss()
    loss = criterion(state_action_values[0], expected_state_action_values[0])
    loss_cpy = loss.detach()

    # Optimize the model
    loss.backward()
    for param in self.policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    self.optimiser.step()

    return loss_cpy

  def policy(self, state, step: int = 0, eps_greedy: bool = True):
    self.policy_net.eval()
    sample = random.random()
    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
      -1.0 * step / self.eps_decay
    )
    self.eps_threshold = eps_threshold
    # wandb.log({'eps_threshold': eps_threshold})
    # args['step'] += 1

    if sample > eps_threshold or not eps_greedy:
    # if False:
      with torch.no_grad():
        output = self.policy_net(state)
        mask = self.mask_unavailable_moves().to(self.device)
        indexes = torch.arange(start=0, end=self.output_size)
        output = output[mask]
        indexes = indexes[mask]
        max_utility = output.max(0)[1].item()
        best_action = indexes[max_utility].item()
        return best_action
    else:
      random_order = self.pm.available_orders[int(random.random() * len(self.pm.available_orders))]
      return self.encode_action(random_order)


  def decode_action(
        self,
        neuron_pos:int,
    ) -> Union[DoubleBattleOrder, DefaultBattleOrder]:
    if neuron_pos < self.n_actions * (self.n_moves * self.n_targets):
      first_order = None
      second_order = None
      # the first one wants to make a move
      act_1 = neuron_pos // self.n_actions # [0, self.n_targets*self.n_moves)
      first_order = self.decode_order(act_1, -1)
      act_2 = neuron_pos % self.n_actions # [0, self.n_actions)
      second_order = self.decode_order(act_2, -2)
      return DoubleBattleOrder(first_order, second_order)
    elif neuron_pos < self.n_actions * (self.n_moves * self.n_targets) + (
      (self.n_actions - 1) * self.n_switches
    ):
      switch = neuron_pos - (self.n_actions * (self.n_moves * self.n_targets))
      switch_1_trg = switch // (self.n_actions - 1) # [0, self.n_switches)
      # search in available switches
      mon_1 = self.pm.available_switches[-1][switch_1_trg]
      first_order = BattleOrder(mon_1)
      act_2 = switch % (self.n_actions - 1) # [0, self.n_actions-1)
      if act_2 < (self.n_moves * self.n_targets): # i.e. a move
        second_order = self.decode_order(act_2, -2)
      else:
        possible_switches = [
          i for i in range(0, self.n_switches) if i != switch_1_trg
        ] # a list with only the possible target switches
        switch_2_trg = possible_switches[
          act_2 - (self.n_moves * self.n_targets)
        ] # [0, self.n_switches-1), but removing the same
        mon_2 = self.pm.available_switches[-2][switch_2_trg]
        second_order = BattleOrder(mon_2)
      return DoubleBattleOrder(first_order, second_order)
    elif (
      neuron_pos
      < self.n_actions * (self.n_moves * self.n_targets)
      + ((self.n_actions - 1) * self.n_switches)
      + 2 * self.n_actions
    ):
      action = neuron_pos - (
        self.n_actions * (self.n_moves * self.n_targets) + ((self.n_actions - 1) * self.n_switches)
      )
      act = action % self.n_actions
      if action // self.n_actions == 0:
        # first pokemon
        first_order = self.decode_order(act, -1)
      else:
        # second pokemon
        first_order = self.decode_order(act, -2)
      return DoubleBattleOrder(first_order, None)
    else:
      return DefaultBattleOrder()


  def decode_order(
    self,
    act: int, pos: int
  ) -> BattleOrder:
    if act < (self.n_moves * self.n_targets):
      # the pokemon wants to perform a move
      move_idx = act // self.n_targets
      move_trg = act % self.n_targets
      move = self.pm.available_moves[pos][move_idx]
      order = BattleOrder(move, move_target=move_trg - 2)
    else:
      # the pokemon wants to perform a switch
      switch_trg = act - (self.n_moves * self.n_targets)
      mon = self.pm.available_switches[pos][switch_trg]
      order = BattleOrder(mon)
    return order


  def encode_action(
    self,
    order: Union[DoubleBattleOrder, DefaultBattleOrder],
  ) -> int:
    idx = 0
    if isinstance(order, DefaultBattleOrder):
      idx += (
        self.n_actions * (self.n_moves * self.n_targets)
        + (self.n_actions - 1) * self.n_switches
        + 2 * self.n_actions
      )
    else:
      first_order = order.first_order
      second_order = order.second_order
      if second_order is None:
        idx += self.n_actions * (self.n_moves * self.n_targets) + (self.n_actions - 1) * self.n_switches
        if (-1 in self.current_battle.available_switches and self.current_battle.force_switch[0]) or self.current_battle.active_pokemon[0]:
          idx += self.encode_order(first_order, -1)
        elif (-2 in self.current_battle.available_switches and self.current_battle.force_switch[1]) or self.current_battle.active_pokemon[1]:
          idx += self.n_actions
          idx += self.encode_order(first_order, -2)
        else:
          import pdb

          pdb.set_trace()
      else:
        first_idx = self.encode_order(first_order, -1)
        second_idx = self.encode_order(second_order, -2)

        if first_idx < self.n_moves * self.n_targets:
          idx = first_idx * (self.n_actions)
          idx += second_idx
        else:
          idx = self.n_actions * (self.n_moves * self.n_targets)
          idx += (self.n_actions - 1) * (first_idx - (self.n_moves * self.n_targets))
          idx += second_idx
          # given that some switches are not possible, we collapse them into a single one
          if second_idx > first_idx:
            idx -= 1
    return idx


  def encode_order(
    self,
    order: BattleOrder, pos: int
  ) -> int:
    if isinstance(order.order, OriginalMove):
      target = order.move_target
      target_idx = target + 2
      move_idx = self.pm.available_moves[pos].index(Move(order.order._id))
      return move_idx * self.n_targets + target_idx
    elif isinstance(order.order, Pokemon):
      try:
        switches = self.pm.available_switches[pos]
      except:
        import pdb; pdb.set_trace()
      for i, mon in enumerate(switches):
        if mon.species == order.order.species:
          return self.n_targets * self.n_moves + i


  def mask_unavailable_moves(self) -> torch.Tensor:
    mask = torch.zeros(self.output_size, dtype=torch.bool)
    if self.pm.available_orders:
      for order in self.pm.available_orders:
        idx = self.encode_action(order)
        if mask[idx] == False:
          mask[idx] = True
        else:
          import pdb

          print("error in the mapping or same battle order")

          pdb.set_trace()
    else:
      idx = self.encode_action(DefaultBattleOrder())
      mask[idx] = True
    return mask



class ParetoRLPLayer(CombineActionRLPlayer):
  def __init__(
      self,
      input_size: int,
      hidden_layers: List[int],
      n_switches: int,
      n_moves: int,
      n_targets: int,
      eps_start: float,
      eps_end: float,
      eps_decay: float,
      batch_size: int,
      gamma: float,
      **kwargs
      ):
    super(ParetoRLPLayer, self).__init__(input_size,hidden_layers,n_switches,n_moves,n_targets,eps_start,eps_end,eps_decay,batch_size,gamma,**kwargs)
    self.agent = AsyncParetoPlayer(
      user_funcs=self,
      username=self.__class__.__name__,
      **kwargs
    )
    # self.actions = self.agent.actions
    # self.observations = self.agent.observations

  def policy(self, state, step: int = 0, eps_greedy: bool = True):
    self.policy_net.eval()
    sample = random.random()
    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
      -1.0 * step / self.eps_decay
    )
    self.eps_threshold = eps_threshold
    # wandb.log({'eps_threshold': eps_threshold})
    # args['step'] += 1

    if sample > eps_threshold or not eps_greedy:
    # if False:
      with torch.no_grad():
        output = self.policy_net(state)
        mask = self.mask_unavailable_moves().to(self.device)
        indexes = torch.arange(start=0, end=self.output_size)
        output = output[mask]
        indexes = indexes[mask]
        max_utility = output.max(0)[1].item()
        best_action = indexes[max_utility].item()
        return best_action
    else:
      random_order = self.pm.available_orders[int(random.random() * len(self.pm.available_orders))]
      return self.encode_action(random_order)


TEAM = """
Zacian-Crowned @ Rusted Sword
Ability: Intrepid Sword
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Substitute
- Behemoth Blade
- Sacred Sword
- Protect

Landorus-Therian (M) @ Sitrus Berry
Ability: Intimidate
EVs: 68 HP / 252 Atk / 4 SpD / 184 Spe
Jolly Nature
- Rock Slide
- Earthquake
- Protect
- Swords Dance

Groudon @ Assault Vest
Ability: Drought
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Rock Slide
- Fire Punch
- Precipice Blades
- Dragon Claw

Charizard @ Life Orb
Ability: Solar Power
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Air Slash
- Protect
- Solar Beam
- Heat Wave

Venusaur @ Coba Berry
Ability: Chlorophyll
EVs: 180 HP / 76 SpA / 252 Spe
IVs: 0 Atk
- Protect
- Sleep Powder
- Leaf Storm
- Earth Power

Urshifu-Rapid-Strike @ Focus Sash
Ability: Unseen Fist
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Detect
- Aqua Jet
- Close Combat
- Surging Strikes
"""

TEAM = """
Zacian-Crowned @ Rusted Sword
Ability: Intrepid Sword
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Substitute
- Behemoth Blade
- Sacred Sword
- Protect

Venusaur @ Coba Berry
Ability: Chlorophyll
EVs: 180 HP / 76 SpA / 252 Spe
IVs: 0 Atk
- Protect
- Sleep Powder
- Leaf Storm
- Earth Power

Groudon @ Assault Vest
Ability: Drought
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Rock Slide
- Fire Punch
- Precipice Blades
- Dragon Claw
"""

TEAM = """
Zacian-Crowned @ Rusted Sword
Ability: Intrepid Sword
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Substitute
- Behemoth Blade
- Sacred Sword
- Protect

Venusaur @ Coba Berry
Ability: Chlorophyll
EVs: 180 HP / 76 SpA / 252 Spe
IVs: 0 Atk
- Protect
- Sleep Powder
- Leaf Storm
- Earth Power
"""
