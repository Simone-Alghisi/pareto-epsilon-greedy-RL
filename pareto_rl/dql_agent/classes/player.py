import torch
import random
import math
import torch.nn as nn
import torch.optim as optim
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from gym.spaces import Space
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder
from poke_env.environment.battle import Battle
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from typing import Dict, List
from pareto_rl.dql_agent.classes.darkr_ai import DarkrAI, Transition, ReplayMemory
from abc import ABC, abstractmethod

class SimpleRLPlayer(Gen8EnvSinglePlayer):
  def __init__(self, **kwargs):
    super(SimpleRLPlayer, self).__init__(**kwargs)

  def embed_battle(self, battle: DoubleBattle) -> list:
    obs = []
    active = []
    bench = []

    for mon in battle.team.values():
      # lots of info are available, the problem is time,
      # hughes effect, and also mapping
      mon_data = []

      # types (2)
      types = [t.value if t is not None else -1 for t in mon.types]
      mon_data.extend(types)

      # hp normalised (good idea?)
      mon_data.append(mon.current_hp_fraction)

      # stats (5)
      mon_data.extend(list(mon.stats.values()))

      # boosts and debuffs (7)
      # TODO it may be possible to compute it together with
      # the stats above to reduce the parameters
      mon_data.extend(list(mon.boosts.values()))

      # status
      # TODO one-hot-encoding?
      mon_data.append(mon.status.value if mon.status is not None else -1)

      # moves
      # TODO... is it possible to have less than 4?
      for move in mon.moves.values():
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
        move_data.append(move_damage)

        # priority
        move_data.append(move.priority)

        # accuracy
        # TODO... should we encode together w/ damage?
        move_data.append(move.accuracy)

        # category (?)
        move_data.append(move.category.value)

        # pp (?)
        # move_data.append(move.current_pp / move.max_pp)

        # recoil (?)
        # move_data.append(move.recoil)

        # damage for each active opponent (2)
        for opp in battle.opponent_active_pokemon:
          if opp is not None:
            mlt = move.type.damage_multiplier(opp.type_1, opp.type_2)
            move_data.append(move_damage*mlt)
          else:
            # if one is dead, append -1
            move_data.append(-1)

        mon_data.extend(move_data)

      mon_data.extend([-1 for _ in range(6)]*(4-len(mon.moves)))

      if mon.active == True:
        active.extend(mon_data)
      else:
        bench.extend(mon_data)

    obs.extend(active)
    obs.extend(bench)

    # we could also take into account the opponents
    # (which I would say is mandatory)
    # and the field conditions (at least some of them)
    return obs

  def describe_embedding(self) -> Space:
    return super().describe_embedding()

  def calc_reward(self, last_battle, current_battle) -> float:
    return self.reward_computing_helper(current_battle,fainted_value=2,hp_value=1,victory_value=30)

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
    super(BaseRLPlayer, self).__init__(**kwargs)
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

  def _init_model(self, input_size, hidden_layers):
    self.policy_net = DarkrAI(input_size, hidden_layers, self.n_actions).to(self.device)
    self.target_net = DarkrAI(input_size, hidden_layers, self.n_actions).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimiser = optim.Adam(self.policy_net.parameters())

  @abstractmethod
  def optimize_model(self, memory: ReplayMemory):
    pass

  @abstractmethod
  def policy(self, state, step: int = 0, eps_greedy: bool = True):
    pass

  @abstractmethod
  def update_target(self):
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
    self.n_actions = (n_moves*n_targets + n_switches)*2
    self._init_model(input_size,hidden_layers)

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
      if switch >= len(battle.available_switches[idx]):
        import pdb; pdb.set_trace()
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

  def _is_valid(self, pos:int, encoded_move_idx: int, poke_mapper: PokemonMapper):
    valid = False
    n_targets = self.n_targets
    mon_actions = self.n_actions // 2
    if pos in poke_mapper.original_moves_targets:
      #TODO check if order is the same here and in rlplayer
      moves = [targets for _, targets in poke_mapper.original_moves_targets[pos].items()]

      if encoded_move_idx >= mon_actions-4:
        # check validity of switch
        # TODO check if benched pokemons' order/position matters
        n_alive = len([mon for mon in self.current_battle.team.values() if not mon.fainted])
        if n_alive > 2:
          max_switch = n_alive - 2
          switch_target = encoded_move_idx - (mon_actions - 4)
          if switch_target < max_switch:
            if pos in poke_mapper.available_switches and len(poke_mapper.available_switches[pos]) > 0:
              valid = True
      else:
        # check validity of attack
        move_idx = encoded_move_idx // n_targets
        target = (encoded_move_idx % n_targets) - 2
        if move_idx < len(moves):
          if target in moves[move_idx]:
            valid = True
    return valid


  def _get_valid_actions(self, poke_mapper: PokemonMapper, actions: torch.Tensor, utilities: torch.Tensor):
    valid_moves = False
    move = [0,0]
    mon_actions = self.n_actions // 2
    ally_pos = [pos for pos in poke_mapper.pos_to_mon.keys() if pos < 0]
    pos_to_idx = {-1: 0, -2: 1}
    # TODO consider dead pokemon
    while not valid_moves:
      # get first valid actions for both players
      for pos in ally_pos:
        # mon = -2 if mon == 0 else -1
        idx = pos_to_idx[pos]
        while (
            move[idx] < mon_actions and
            not self._is_valid(pos, actions[idx][move[idx]].item(), poke_mapper)
        ):
          move[idx] += 1

      if move[0] < mon_actions and move[1] < mon_actions:
        # check if both moves are switches and if they are the same
        if(actions[0][move[0]] >= (mon_actions-4) and actions[1][move[1]] >= (mon_actions-4)):
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

    poke_mapper = PokemonMapper(self.current_battle)
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
        return self._get_valid_actions(poke_mapper,actions,utilities)
    else:
      # TODO maybe more efficient getting random valid action then decoding
      mon_actions = self.n_actions // 2
      actions = torch.stack([torch.randperm(mon_actions) for _ in range(2)])
      utilities = torch.rand(2,mon_actions)
      return self._get_valid_actions(poke_mapper,actions,utilities)

  def _encode_actions(self, actions):
    return actions[0]*100 + actions[1]

  def _decode_actions(self, coded_action):
    action1 = coded_action//100
    action2 = coded_action % 100
    return [action1,action2]

  def update_target(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

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
    # TODO define proper n_actions
    self.n_actions = 0
    self._init_model(input_size,hidden_layers)

  def action_to_move(self, actions, battle: Battle) -> BattleOrder:  # pyre-ignore
    return ForfeitBattleOrder()

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
    pass

  def update_target(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

