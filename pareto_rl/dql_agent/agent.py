from logging import info
import torch
import torch.optim as optim
import torch.nn as nn
import math
import random
from itertools import count
from pareto_rl.dql_agent.classes.darkr_ai import DarkrAI, Transition, ReplayMemory
from pareto_rl.dql_agent.classes.player import SimpleRLPlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
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
  transitions = memory.sample(args['batch_size'])
  # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
  # detailed explanation). This converts batch-array of Transitions
  # to Transition of batch-arrays.
  batch = Transition(*zip(*transitions))

  # Compute a mask of non-final states and concatenate the batch elements
  # (a final state would've been the one after which simulation ended)
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=args['device'], dtype=torch.bool)
  non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
  print(batch.action)
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)

  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  state_action_values = policy_net(state_batch).gather(1, action_batch)

  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.
  next_state_values = torch.zeros(args['batch_size'], device=args['device'])
  next_state_values[non_final_mask] = target_net(non_final_next_states).argmax().detach()
  # Compute the expected Q values
  expected_state_action_values = (next_state_values * args['gamma']) + reward_batch

  # Compute Huber loss
  criterion = nn.SmoothL1Loss()
  loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

  # Optimize the model
  optimiser.zero_grad()
  loss.backward()
  # for param in policy_net.parameters():
  #   param.grad.data.clamp_(-1, 1)
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
      return policy_net(state).argmax().detach().squeeze()
  else:
    return torch.tensor([[random.randrange(args['n_actions'])]], device=args['device'], dtype=torch.long)

def get_state():
  pass

def get_reward():
  pass

def test_alg(player: SimpleRLPlayer):

  player.reset()
  # player.reset_battles()
  print(player._actions)

  action = player.action_space[0]

  observation,reward,done,info = player.step(action)
  print(observation)
  print(reward)
  print(info)
  # player.complete_current_battle()

def train(player,**args):

  hidden_layers = [32, 16]
  n_actions = len(player.action_space)
  args['n_actions'] = n_actions
  input_size = 10
  policy_net = DarkrAI(input_size, hidden_layers, n_actions).to(args['device'])
  target_net = DarkrAI(input_size, hidden_layers, n_actions).to(args['device'])
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  optimiser = optim.Adam(policy_net.parameters())
  memory = ReplayMemory(30)

  episode_durations = []


  # train loop
  num_episodes = 50
  for i_episode in range(num_episodes):
    # games
    # Initialize the environment and state

    observation = torch.from_numpy(player.reset()).double().to(args['device'])
    state = observation

    for t in count():
      # turns
      # Select and perform an action
      action = policy(state, policy_net, args)
      print(action)
      observation, reward, done, _ = player.step(action)
      observation = torch.from_numpy(observation).double().to(args['device'])
      print(reward)

      reward = torch.tensor([reward], device=args['device'])

      # Observe new state
      if not done:
        next_state = observation
      else:
        next_state = None

      # Store the transition in memory
      memory.push(state, action, next_state, reward)

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

def main(args):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  args = {
    'batch_size':16,
    'gamma': 0.999,
    'target_update': 10,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 200,
    'device': device,
    'steps': 0,
  }

  darkrai_player_config = PlayerConfiguration("DarkrAI", None)
  random_player_config = PlayerConfiguration("RandomOpponent",None)

  env_player = SimpleRLPlayer(battle_format="gen8randombattle",player_configuration=darkrai_player_config)
  opponent = RandomPlayer(battle_format="gen8randombattle",player_configuration=random_player_config)
  env_player.play_against(
    env_algorithm=train,
    opponent=opponent,
    env_algorithm_kwargs=args
  )

