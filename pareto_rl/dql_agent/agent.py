from logging import info
import torch
import torch.optim as optim
import torch.nn as nn
import math
import random
import os
from tqdm import tqdm
from itertools import count
from pareto_rl.dql_agent.classes.darkr_ai import DarkrAI, Transition, ReplayMemory
from pareto_rl.dql_agent.classes.pareto_player import ParetoPlayer
from pareto_rl.dql_agent.classes.player import SimpleRLPlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from typing import Dict, List

def configure_subparsers(subparsers):
  r"""Configure a new subparser for DQL agent.

  Args:
    subparsers: subparser
  """

  """
  Subparser parameters:
  Args:

  """
  parser = subparsers.add_parser("rlagent", help="Train/test reinforcement learning")
  parser.set_defaults(func=main)

def optimize_model(memory, policy_net, target_net, optimiser, args):
  if len(memory) < args['batch_size']:
    return
  policy_net.train()
  policy_net.zero_grad()
  optimiser.zero_grad()
  transitions = memory.sample(args['batch_size'])
  # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
  # detailed explanation). This converts batch-array of Transitions
  # to Transition of batch-arrays.
  batch = Transition(*zip(*transitions))

  # Compute a mask of non-final states and concatenate the batch elements
  # (a final state would've been the one after which simulation ended)
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=args['device'], dtype=torch.bool)
  non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

  # state_batch = torch.cat(batch.state)
  state_batch = torch.stack(batch.state)
  # action_batch = torch.cat(batch.action)
  action_batch = [[],[]]
  for action in batch.action:
    action_batch[0].append(action[0])
    action_batch[1].append(action[1])
  action_batch = torch.tensor(action_batch, device=args['device'])
  reward_batch = torch.cat(batch.reward).double()

  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  utility = policy_net(state_batch)
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
      torch.zeros(args['batch_size'], dtype=torch.float64, device=args['device']),
      torch.zeros(args['batch_size'], dtype=torch.float64, device=args['device'])
  ]
  # Select greedily max action (off-policy, Q-Learning)
  next_state_utility = target_net(non_final_next_states)
  # Split for first and second player actions
  next_state_utility = torch.split(next_state_utility, split_size_or_sections=next_state_utility.shape[1] // 2, dim=1)
  next_state_values[0][non_final_mask] = next_state_utility[0].max(1)[0]
  next_state_values[1][non_final_mask] = next_state_utility[1].max(1)[0]

  # Compute the expected Q values
  expected_state_action_values = [
      (next_state_values[0] * args['gamma']) + reward_batch,
      (next_state_values[1] * args['gamma']) + reward_batch
  ]

  # Compute Huber loss
  criterion = nn.HuberLoss()
  loss = criterion(state_action_values[0], expected_state_action_values[0]) \
       + criterion(state_action_values[1], expected_state_action_values[1])

  # Optimize the model
  loss.backward()
  optimiser.step()


def is_valid(pos:int, encoded_move_idx: int, poke_mapper: PokemonMapper, battle: DoubleBattle, args):
  # TODO pokemon to pos
  valid = False
  n_targets = args['n_targets']
  mon_actions = args['n_actions'] // 2
  if pos in poke_mapper.original_moves_targets:
    #TODO check if order is the same here and in rlplayer
    # print(poke_mapper.original_moves_targets[pos])
    moves = [targets for _, targets in poke_mapper.original_moves_targets[pos].items()]
    # print(moves)

    if encoded_move_idx >= mon_actions-4:
      # check validity of switch
      # TODO check if benched pokemons' order/position matters
      n_alive = len([mon for mon in battle.team.values() if not mon.fainted])
      if n_alive > 2:
        max_switch = n_alive - 2
        switch_target = encoded_move_idx - (mon_actions - 4)
        if switch_target < max_switch:
          if pos in poke_mapper.available_switches and len(poke_mapper.available_switches[pos]) > 0:
            valid = True
            # print(f'Encoded: {encoded_move_idx}')
            # print(f'Valid-switch: {switch_target}')
    else:
      # check validity of attack
      move_idx = encoded_move_idx // n_targets
      target = (encoded_move_idx % n_targets) - 2
      if move_idx < len(moves):
        if target in moves[move_idx]:
          valid = True
          # print(f'Encoded: {encoded_move_idx}')
          # print(f'Valid: {move_idx} {target}')
  return valid


def get_valid_actions(poke_mapper: PokemonMapper, battle: DoubleBattle, actions: torch.Tensor, utilities: torch.Tensor, args):
  valid_moves = False
  move = [0,0]
  mon_actions = args['n_actions'] // 2
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
          not is_valid(pos, actions[idx][move[idx]].item(), poke_mapper, battle, args)
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
    # print(torch.tensor([ actions[idx][move[idx]].item() for idx in [0,1] ]).tolist())
    return torch.tensor([ actions[idx][move[idx]] for idx in [0,1] ])

def policy(state, policy_net, battle: DoubleBattle, args, eps_greedy: bool = True):
  policy_net.eval()
  sample = random.random()
  eps_threshold = args['eps_end'] + (args['eps_start'] - args['eps_end']) * math.exp(-1. * args['episodes'] / args['eps_decay'])
  # args['steps'] += 1

  poke_mapper = PokemonMapper(battle)
  if sample > eps_threshold or not eps_greedy:
  # if True:
    with torch.no_grad():
      #divide output in 2 halves, select the max in the first half and the max in the second half.
      output = policy_net(state)
      split_index = len(output) // 2
      # order actions and utilities
      outputs = [output[:split_index].sort(descending=True), output[split_index:].sort(descending=True)]
      utilities = torch.stack([ tensor.values for tensor in outputs ])
      actions = torch.stack([ tensor.indices for tensor in outputs ])
      return get_valid_actions(poke_mapper,battle,actions,utilities,args)
  else:
    # TODO maybe more efficient getting random valid action then decoding
    mon_actions = args['n_actions'] // 2
    actions = torch.stack([torch.randperm(mon_actions) for _ in range(2)])
    utilities = torch.rand(2,mon_actions)
    return get_valid_actions(poke_mapper,battle,actions,utilities,args)

def encode_actions(actions):
  return actions[0]*100 + actions[1]

def does_anybody_have_tabu_moves(battle: DoubleBattle, tabus: List[str]):
  for mon in battle.team.values():
    if mon:
      for move in mon.moves.values():
        if move._id in tabus:
          return True
  return False

def is_anyone_someone(battle: DoubleBattle, monsters: List[str]):
  for mon in battle.team.values():
    if mon:
      if mon.species in monsters:
        return True
  return False

def train(player:SimpleRLPlayer, num_episodes: int, args):
  policy_net = DarkrAI(args['input_size'], args['hidden_layers'], args['n_actions']).to(args['device'])
  target_net = DarkrAI(args['input_size'], args['hidden_layers'], args['n_actions']).to(args['device'])
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()
  optimiser = optim.Adam(policy_net.parameters())
  memory = ReplayMemory(10000)
  eval_config = PlayerConfiguration("DarkrAI_eval",None)
  eval_player = SimpleRLPlayer(battle_format="gen8randomdoublesbattle",player_configuration=eval_config)

  episode_durations = []

  # train loop
  for i_episode in tqdm(range(num_episodes), desc='Training', unit='episodes'):
    # games
    args['episodes'] = i_episode

    # Intermediate evaluation
    if i_episode > 0 and i_episode % args['eval_interval'] == 0:
      eval(eval_player,args['periodic_eval_games'],policy_net=policy_net,**args)

    # Initialize the environment and state
    observation = torch.tensor(player.reset(), dtype=torch.double, device=args['device'])
    prev_state = state = observation

    if does_anybody_have_tabu_moves(player.current_battle, ['transform', 'allyswitch']):
      print('Damn you, \nMew!\nAnd to all that can AllySwitch\nDamn you, \ntoo!')
      # TODO force finish game?
      continue
    if is_anyone_someone(player.current_battle, ['ditto', 'zoroark']):
      print('Damn you three, \nDitto and Zoroark!')
      continue


    for t in count():
      # turns

      if state.shape[0] < 240:
        import pdb; pdb.set_trace()
      # Select and perform an action
      actions = policy(state, policy_net, player.current_battle, args)
      # print(f'Move {t}: {player.action_to_move(encode_actions(actions.tolist()), player.current_battle)}')
      # if torch.equal(prev_state,state) and t > 0:
      #   import pdb; pdb.set_trace()

      observation, reward, done, _ = player.step(encode_actions(actions.tolist()))
      observation = torch.tensor(observation, dtype=torch.double, device=args['device'])
      reward = torch.tensor([reward], device=args['device'])

      # Observe new state
      if not done:
        next_state = observation
      else:
        next_state = None

      # Store the transition in memory
      memory.push(state, actions, next_state, reward)
      # Move to the next state
      prev_state = state
      state = next_state

      # Perform one step of the optimization (on the policy network)
      optimize_model(memory, policy_net, target_net, optimiser, args)
      if done:
        episode_durations.append(t + 1)
        break
    # Update the target network, copying all weights and biases in DQN
    if i_episode > 0 and i_episode % args['target_update'] == 0:
      target_net.load_state_dict(policy_net.state_dict())
  # print(episode_durations)
  #player.complete_current_battle()
  player.reset_env()
  model_path = os.path.abspath('./models/best.pth')
  torch.save(policy_net.state_dict(), model_path)


def eval(player: SimpleRLPlayer, num_episodes: int, policy_net=None, **args):
  if policy_net == None:
    policy_net = DarkrAI(args['input_size'], args['hidden_layers'], args['n_actions']).to(args['device'])
    model_path = os.path.abspath('./models/best.pth')
    if os.path.exists(model_path):
      policy_net.load_state_dict(torch.load(model_path))
    else:
      print(f'Error: No model found for evaluation')
      return
  policy_net.eval()

  if player.current_battle is not None:
    player.reset_env(restart=False)
    player.reset_battles()
    player.start_challenging()

  episode_durations = []
  for _ in tqdm(range(num_episodes), desc='Evaluating', unit='episodes'):
    observation = torch.tensor(player.reset(), dtype=torch.double, device=args['device'])
    state = observation

    if does_anybody_have_tabu_moves(player.current_battle, ['transform', 'allyswitch']):
      print('Damn you, \nMew!\nAnd to all that can AllySwitch\nDamn you, \ntoo!')
      # TODO force finish game?
      continue
    if is_anyone_someone(player.current_battle, ['ditto', 'zoroark']):
      print('Damn you three, \nDitto and Zoroark!')
      continue

    for t in count():
      # action = policy(state, policy_net, args)
      # Follow learned policy (eps_greedy=False -> never choose random move)
      actions = policy(state, policy_net, player.current_battle, args, eps_greedy=False)

      observation, _, done, _ = player.step(encode_actions(actions))
      observation = torch.tensor(observation, dtype=torch.double, device=args['device'])

      # Observe new state
      if not done:
        next_state = observation
      else:
        episode_durations.append(t + 1)
        break

      # Move to the next state
      state = next_state

  # print(episode_durations)
  # player.complete_current_battle()
  # player.reset_env()
  print(f'DarkrAI has won {player.n_won_battles} out of {num_episodes} games')

def main(args):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  hidden_layers = [180, 120]
  n_moves = 4
  n_switches = 4
  n_targets = 5
  n_actions = (n_moves*n_targets + n_switches)*2
  input_size = 240
  args = {
    'batch_size': 128,
    'gamma': 0.999,
    'target_update': 10,
    'eval_interval': 500,
    'periodic_eval_games': 100,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 200,
    'device': device,
    'episodes': 0,
    'n_moves': n_moves,
    'n_targets': n_targets,
    'n_actions': n_actions,
    'input_size': input_size,
    'hidden_layers': hidden_layers,
  }

  darkrai_player_config = PlayerConfiguration("DarkrAI",None)

  agent=SimpleRLPlayer(battle_format="gen8randomdoublesbattle",player_configuration=darkrai_player_config)
  train(agent,5000,args)
  eval(agent,1000,**args)
