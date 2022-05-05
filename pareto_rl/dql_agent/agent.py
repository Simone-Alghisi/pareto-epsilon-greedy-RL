import asyncio
from logging import info
from poke_env.player import random_player
from poke_env.player.openai_api import OpenAIGymEnv
import torch
import torch.optim as optim
import torch.nn as nn
import math
import random
import os
from tqdm import tqdm
from itertools import count
from pareto_rl.dql_agent.classes import player
from pareto_rl.dql_agent.classes.darkr_ai import DarkrAI, Transition, ReplayMemory
from pareto_rl.dql_agent.classes.pareto_player import ParetoPlayer
from pareto_rl.dql_agent.classes.player import SimpleRLPlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.player.player import Player
from typing import List
from poke_env.player.battle_order import (
    BattleOrder,
)
import orjson
from poke_env.environment.double_battle import DoubleBattle

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
  action_batch = torch.tensor(batch.action, device=args['device'])
  reward_batch = torch.cat(batch.reward).double()

  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  utility = policy_net(state_batch)
  # For each selected action, select its corresponding utility value
  state_action_values = utility.gather(1,action_batch.unsqueeze(1)).squeeze()

  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.
  next_state_values = torch.zeros(args['batch_size'], dtype=torch.float64, device=args['device'])
  # Select greedily max action (off-policy, Q-Learning)
  next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
  # Compute the expected Q values
  expected_state_action_values = (next_state_values * args['gamma']) + reward_batch

  # Compute Huber loss
  # criterion = nn.HuberLoss()
  criterion = nn.HuberLoss()
  loss = criterion(state_action_values, expected_state_action_values)
  print(loss)

  # Optimize the model
  loss.backward()
  optimiser.step()

def policy(state, policy_net, args):
  sample = random.random()
  eps_threshold = args['eps_end'] + (args['eps_start'] - args['eps_end']) * math.exp(-1. * args['steps'] / args['eps_decay'])
  args['steps'] += 1

  if sample > eps_threshold:

     with torch.no_grad():
       # t.max(1) will return largest column value of each row.
       # second column on max result is index of where max element was
       # found, so we pick action with the larger expected reward.
       print("||||||||||||||||| Mossa policy ||||||||||||||||||")
       #divide output in 2 halves, select the max in the first half and the max in the second half.
       output = policy_net(state).detach()
       halfIndex = len(output) // 2
       first_utility,first_action = output[:halfIndex].topk(2)
       second_utility,second_action = output[halfIndex:].topk(2)
       selected_first_action = first_action[0]
       selected_second_action = second_action[0]
       if(first_action[0] > 8 and second_action[0] > 8):
         #both switch actions
         if(first_action[0] == second_action[0]):
           if(first_utility[0] > second_utility[0]):
             #get second for second utility
             selected_second_action = second_action[1]
           else:
             selected_first_action = first_action[1]




       print(selected_first_action,selected_second_action)
       return torch.cat([selected_first_action.unsqueeze(0),selected_second_action.unsqueeze(0)],dim=0)
  else:
    actions = [i for i in range(args['n_actions']//2)]
    action1 = random.choice(actions)
    if(action1 >= 8):
      actions.remove(action1)
    action2 = random.choice(actions)
    random_actions = torch.tensor([action1,action2],device=args['device'], dtype=torch.long)
    return random_actions

def encode_actions(actions):
  return actions[0]*100 + actions[1]

def train(player:SimpleRLPlayer, args):
  hidden_layers = [32, 16]
  print(args)
  print(player.action_space)

  n_actions = 24
  args['n_actions'] = n_actions
  input_size = 26
  policy_net = DarkrAI(input_size, hidden_layers, n_actions).to(args['device'])
  target_net = DarkrAI(input_size, hidden_layers, n_actions).to(args['device'])
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()
  optimiser = optim.Adam(policy_net.parameters())
  memory = ReplayMemory(10000)

  episode_durations = []

  # train loop
  num_episodes = 5000
  for i_episode in tqdm(range(num_episodes), desc='Training', unit='episodes'):
    # games
    # Initialize the environment and state

    observation = torch.from_numpy(player.reset()).double().to(args['device'])
    state = observation
    for t in count():
      # turns
      # Select and perform an action

      actions = policy(state, policy_net, args)

      # for i in range(26):
      #     print(i)
      #     print(type(player.action_to_move(i,player.current_battle)))
      #     print(f"|||||||||||| {player.current_battle.available_moves} |||||||||||||||")
      #     print(f"|||||||||||| {player.current_battle.available_switches} ||||||||||||||")
      #     print(f"|||||||||||| {player.action_to_move(i,player.current_battle)} ||||||||||||||||")
      #
      #
      # import pdb; pdb.set_trace()

      observation, reward, done, _ = player.step(encode_actions(actions))
      observation = torch.from_numpy(observation).double().to(args['device'])
      reward = torch.tensor([reward], device=args['device'])

      # Observe new state
      if not done:
        next_state = observation
      else:
        next_state = None

      # Store the transition in memory
      memory.push(state, actions, next_state, reward)
      # Move to the next state
      state = next_state

      # Perform one step of the optimization (on the policy network)
      optimize_model(memory, policy_net, target_net, optimiser, args)
      if done:
        episode_durations.append(t + 1)
        break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % args['target_update'] == 0:
      target_net.load_state_dict(policy_net.state_dict())
  print(episode_durations)
  #player.complete_current_battle()
  model_path = os.path.abspath('./models/best.pth')
  torch.save(policy_net.state_dict(), model_path)


def eval(player: SimpleRLPlayer, **args):
  hidden_layers = [32, 16]
  n_actions = player.action_space_size()
  args['n_actions'] = n_actions
  input_size = 26
  policy_net = DarkrAI(input_size, hidden_layers, n_actions).to(args['device'])


  model_path = os.path.abspath('./models/best.pth')
  if os.path.exists(model_path):
    policy_net.load_state_dict(torch.load(model_path))
  else:
    print(f'Error: No model found for evaluation')
    return
  policy_net.eval()

  player.reset_battles()

  episode_durations = []
  num_episodes = 1000
  for _ in tqdm(range(num_episodes), desc='Evaluating', unit='episodes'):
    observation = torch.from_numpy(player.reset()).double().to(args['device'])
    state = observation
    for t in count():
      # action = policy(state, policy_net, args)
      # Follow learned policy
      action = policy_net(state).argmax()
      observation, _, done, _ = player.step(action)
      observation = torch.from_numpy(observation).double().to(args['device'])

      # Observe new state
      if not done:
        next_state = observation
      else:
        episode_durations.append(t + 1)
        break

      # Move to the next state
      state = next_state

  print(episode_durations)
  #player.complete_current_battle()
  print(f'DarkrAI has won {player.n_won_battles} out of {num_episodes} games')

def main(args):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  args = {
    'batch_size': 128,
    'gamma': 0.999,
    'target_update': 10,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 200,
    'device': device,
    'steps': 0,
  }

  darkrai_player_config = PlayerConfiguration("DarkrAI",None)

  agent=SimpleRLPlayer(battle_format="gen8randomdoublesbattle",player_configuration=darkrai_player_config)
  train(agent,args)
