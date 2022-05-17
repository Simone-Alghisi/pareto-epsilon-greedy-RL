from logging import info
import torch
import torch.optim as optim
import torch.nn as nn
import math
import random
import os
import wandb
import pickle
import pandas as pd
from tqdm import tqdm
from itertools import count
from pareto_rl.dql_agent.classes.darkr_ai import DarkrAI, Transition, ReplayMemory
from pareto_rl.dql_agent.classes.pareto_player import ParetoPlayer
from pareto_rl.dql_agent.classes.player import SimpleRLPlayer, BaseRLPlayer, DoubleActionRLPlayer, CombineActionRLPlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from typing import Dict, List
from pareto_rl.dql_agent.classes.random_player import DoubleRandomPlayer
from sklearn.decomposition import PCA

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

def train(player: BaseRLPlayer, num_episodes: int, args):
  memory = ReplayMemory(args['memory'])

  episode_durations = []

  # train loop
  for i_episode in tqdm(range(num_episodes), desc='Training', unit='episodes'):
    # games
    episode_info = {'episode': i_episode}
    # Intermediate evaluation
    if i_episode % args['eval_interval'] == 0:
      winrate = eval(player,args['eval_interval_episodes'],**args)
      episode_info['winrate'] = winrate

    # Initialize the environment and state
    observation = torch.tensor(player.reset(), dtype=torch.double, device=args['device'])
    prev_state = state = observation

    if does_anybody_have_tabu_moves(player.current_battle, ['transform', 'allyswitch']):
      print('Damn you, \nMew!\nAnd to all that can AllySwitch\nDamn you, \ntoo!')
      # TODO force finish game?
      wandb.log(episode_info)
      continue
    if is_anyone_someone(player.current_battle, ['ditto', 'zoroark']):
      print('Damn you three, \nDitto and Zoroark!')
      wandb.log(episode_info)
      continue

    episode_cumul_loss = 0
    episode_cumul_reward = 0

    for t in count():
      step_info = episode_info
      # turns
      args['step'] += 1

      player.update_pm()
      # Select and perform an action
      action = player.policy(state,args['step'])

      # if not args['combined_actions']:
      if isinstance(player, DoubleActionRLPlayer):
        observation, reward, done, _ = player.step(player._encode_actions(action.tolist()))
      else:
        observation, reward, done, _ = player.step(action)
      observation = torch.tensor(observation, dtype=torch.double, device=args['device'])
      reward = torch.tensor([reward], device=args['device'])
      episode_cumul_reward += reward.item()

      # Observe new state
      if not done:
        next_state = observation
      else:
        next_state = None

      # Store the transition in memory
      memory.push(state, action, next_state, reward)
      # Move to the next state
      prev_state = state
      state = next_state

      # Perform one step of the optimization (on the policy network)
      loss = player.optimize_model(memory)
      if loss is not None:
        episode_cumul_loss += loss.item()

      # log current step
      step_info.update({
        'step': args['step'],
        'loss': loss,
        'eps_threshold': player.eps_threshold,
        'reward': reward
      })
      wandb.log(step_info)

      if done:
        episode_durations.append(t + 1)
        break

    episode_info.update({
      'step': args['step'],
      'ep_loss': episode_cumul_loss/(t+1),
      'ep_reward': episode_cumul_reward/(t+1),
    })
    wandb.log(episode_info)
    # Update the target network, copying all weights and biases in DQN
    if i_episode > 0 and i_episode % args['target_update'] == 0:
      player.update_target()

  player.reset_env()
  model_path = os.path.abspath('./models/best.pth')
  torch.save(player.policy_net.state_dict(), model_path)


def eval(player: BaseRLPlayer, num_episodes: int, **args):
  player.policy_net.eval()

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
      player.update_pm()
      # Follow learned policy (eps_greedy=False -> never choose random move)
      actions = player.policy(state, eps_greedy=False)

      if isinstance(player, DoubleActionRLPlayer):
        observation, _, done, _ = player.step(player._encode_actions(actions.tolist()))
      else:
        observation, _, done, _ = player.step(actions)
      observation = torch.tensor(observation, dtype=torch.double, device=args['device'])

      # Observe new state
      if not done:
        next_state = observation
      else:
        episode_durations.append(t + 1)
        break

      # Move to the next state
      state = next_state

  print(f'DarkrAI has won {player.n_won_battles} out of {num_episodes} games')
  return player.n_won_battles/num_episodes

def sample_transitions(player: BaseRLPlayer, num_episodes: int, file_name: str, **args):
  player.policy_net.eval()
  observations = []
  for i_episode in tqdm(range(num_episodes), desc='Evaluating', unit='episodes'):
    observation, labels = player.reset()
    observations.append(observation)
    if i_episode == 0:
      obs_labels = labels
    # for obs, label in zip(observation, labels):
    #   print(f'{label}: {obs}')
    #   print(type(obs))
    observation = torch.tensor(observation, dtype=torch.double, device=args['device'])
    state = observation

    if does_anybody_have_tabu_moves(player.current_battle, ['transform', 'allyswitch']):
      print('Damn you, \nMew!\nAnd to all that can AllySwitch\nDamn you, \ntoo!')
      # TODO force finish game?
      continue
    if is_anyone_someone(player.current_battle, ['ditto', 'zoroark']):
      print('Damn you three, \nDitto and Zoroark!')
      continue

    for t in count():
      player.update_pm()
      # Follow learned policy (eps_greedy=False -> never choose random move)
      actions = player.policy(state, eps_greedy=False)

      if isinstance(player, DoubleActionRLPlayer):
        observation, _, done, _ = player.step(player._encode_actions(actions.tolist()))
      else:
        observation, _, done, _ = player.step(actions)
      observation, _ = observation
      observations.append(observation)
      observation = torch.tensor(observation, dtype=torch.double, device=args['device'])

      # Observe new state
      if not done:
        next_state = observation
      else:
        break

      # Move to the next state
      state = next_state

  data = {
    'labels': labels,
    'observations': observations
  }
  with open('.'.join([file_name,'pickle']), 'wb') as handle:
    pickle.dump(data, handle)

def pca(file_name: str):
  with open('.'.join([file_name,'pickle']), 'rb') as handle:
    data = pickle.load(handle)

  df = pd.DataFrame(data['observations'], columns=data['labels'])
  pca = PCA()
  pca.fit(df)

  pca_df = pd.DataFrame(pca.components_, columns=data['labels'])
  pca_df.to_csv('.'.join([file_name,'csv']))

  # most important features of each new principal component
  components = {
    'max_corr': [],
    'max_feat': []
  }
  for comp in pca.components_:
    max_corr = max(comp)
    idx = comp.tolist().index(max_corr)
    components['max_corr'].append(max_corr)
    components['max_feat'].append(data['labels'][idx])

  most_important_by_pc = pd.DataFrame(components)
  most_important_by_pc.to_csv(''.join([file_name,'most_important','.csv']))

def main(args):
  hidden_layers = [180]
  n_moves = 4
  n_switches = 4
  n_targets = 5
  # input_size = 240
  input_size = 80
  args = {
    'batch_size': 128,
    'gamma': 0.999,
    'target_update': 1500,
    'eval_interval': 500,
    'eval_interval_episodes': 50,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 2*10**4,
    'input_size': input_size,
    'hidden_layers': hidden_layers,
    'train_episodes': 6000,
    'eval_episodes': 100,
    'memory': 10**4,
    'combined_actions': True,
    'fixed_team': True
  }

  battle_format = 'gen8doublesubers' if args['fixed_team'] else 'gen8randomdoublesbattle'

  darkrai_player_config = PlayerConfiguration('DarkrAI',None)
  random_player_config = PlayerConfiguration('RandomMeansRandom',None)
  random_player = DoubleRandomPlayer(battle_format=battle_format, player_configuration=random_player_config)

  if args['combined_actions']:
    agent = CombineActionRLPlayer(
        args['input_size'],
        args['hidden_layers'],
        n_switches,
        n_moves,
        n_targets,
        args['eps_start'],
        args['eps_end'],
        args['eps_decay'],
        args['batch_size'],
        args['gamma'],
        battle_format=battle_format,
        player_configuration=darkrai_player_config,
        opponent=random_player,
        start_timer_on_battle_start=True)
  else:
    agent = DoubleActionRLPlayer(
        args['input_size'],
        args['hidden_layers'],
        n_switches,
        n_moves,
        n_targets,
        args['eps_start'],
        args['eps_end'],
        args['eps_decay'],
        args['batch_size'],
        args['gamma'],
        battle_format=battle_format,
        player_configuration=darkrai_player_config,
        opponent=random_player,
        start_timer_on_battle_start=True)

  # parameters of the run
  args['n_actions'] = agent.n_actions
  args['output_size'] = agent.output_size
  wandb.init(project='DarkrAI', entity='darkr-ai', config=args)

  args.update({
    'device': agent.device,
    'step': 0,
  })

  # remember to change policy so that is always random
  # sample_transitions(agent,1000,'2v2_1k',**args)
  # pca('2v2_1k')

  train(agent,args['train_episodes'],args)
  final_winrate = eval(agent,args['eval_episodes'],**args)
  wandb.log({'winrate': final_winrate})
