import torch
import pickle
import pandas as pd
from logging import info
from tqdm import tqdm
from itertools import count
from pareto_rl.dql_agent.classes.player import BaseRLPlayer, DoubleActionRLPlayer
from typing import Dict, List
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold
from sklearn.cluster import KMeans
from pareto_rl.dql_agent.utils.utils import is_anyone_someone, does_anybody_have_tabu_moves

def sample_transitions(player: BaseRLPlayer, num_episodes: int, file_name: str, **args):
  player.policy_net.eval()
  observations = []
  rewards = []
  obs_labels = []
  for i_episode in tqdm(range(num_episodes), desc='Evaluating', unit='episodes'):
    observation, labels = player.reset()
    # if i_episode == 0:
    if len(labels) > len(obs_labels):
      obs_labels = labels
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
        obs, reward, done, _ = player.step(player._encode_actions(actions.tolist()))
      else:
        obs, reward, done, _ = player.step(actions)
      observation, labels = obs
      observations.append(observation)
      rewards.append(reward)
      if len(labels) > len(obs_labels):
        obs_labels = labels
      observation = torch.tensor(observation, dtype=torch.double, device=args['device'])

      # Observe new state
      if not done:
        next_state = observation
      else:
        break

      # Move to the next state
      state = next_state

  data = {
    'labels': obs_labels,
    'observations': observations,
    'rewards': rewards
  }
  with open(''.join(['./feature_selection/',file_name,'.pickle']), 'wb') as handle:
    pickle.dump(data, handle)

  obs_labels.append('rewards')
  observations = [ obs + [reward] for obs, reward in zip(data['observations'], data['rewards']) ]
  df = pd.DataFrame(observations, columns=obs_labels)
  df.to_csv(''.join(['./feature_selection/',file_name,'.csv']))


def pca(file_name: str, n_components: int, n_most_corr: int):
  with open(''.join(['./feature_selection/',file_name,'.pickle']), 'rb') as handle:
    data = pickle.load(handle)

    df = pd.DataFrame(data['observations'], columns=data['labels'])
    pca = PCA(n_components=n_components)
    pca.fit(df)

    pca_df = pd.DataFrame(pca.components_, columns=data['labels'])
    pca_df.to_csv(''.join(['./feature_selection/',file_name,'_pca','.csv']))

    components = [ {'var_ratio': var_ratio, 'top_k_names': [], 'top_k_corr': []} for var_ratio in pca.explained_variance_ratio_ ]
    feature_scores = {}
    total_score = 0
    for i, correlations in enumerate(pca.components_):
      abs_corr = [ abs(corr) for corr in correlations]
      sorted_corr, labels = zip(*sorted(zip(abs_corr,data['labels']), reverse=True))
      components[i]['top_k_corr'] = sorted_corr[:n_most_corr]
      components[i]['top_k_names'] = labels[:n_most_corr]
      for label, corr in zip(components[i]['top_k_names'], components[i]['top_k_corr']):
        if label not in feature_scores.keys():
          feature_scores[label] = 0
        score = components[i]['var_ratio'] * corr
        total_score += score
        feature_scores[label] += score
    names = [ feat for feat in feature_scores.keys() ]
    scores = [ score/total_score for score in feature_scores.values() ]
    scores, names = zip(*sorted(zip(scores,names),reverse=True))
    feature_scores = {
      'names': names,
      'scores': scores
    }

    df = pd.DataFrame(components)
    df.to_csv(''.join(['./feature_selection/',file_name,'_pca',f'_{n_components}pc_{n_most_corr}corr','.csv']))

    df = pd.DataFrame(feature_scores)
    df.to_csv(''.join(['./feature_selection/',file_name,'_pca','_feature_scores','.csv']))


def sfs(file_name: str):
  with open(''.join(['./feature_selection/',file_name,'.pickle']), 'rb') as handle:
    data = pickle.load(handle)

    df = pd.DataFrame(data['observations'], columns=data['labels'])
    kmeans = KMeans(verbose=False)
    sfs = SequentialFeatureSelector(kmeans, n_features_to_select=20, tol=None, n_jobs=8)
    sfs.fit(df)

    sfs_df = pd.DataFrame(sfs.get_feature_names_out())
    sfs_df.to_csv(''.join(['./feature_selection/',file_name,'_sfs','.csv']))


def variance_threshold(file_name):
  with open(''.join(['./feature_selection/',file_name,'.pickle']), 'rb') as handle:
    data = pickle.load(handle)

    df = pd.DataFrame(data['observations'], columns=data['labels'])
    thresh = 0.8*(1-0.8)
    vt = VarianceThreshold(thresh)
    vt.fit(df)

    thresholded = {
      'variance': [],
      'feature': []
    }
    for var, feat in zip(vt.variances_, data['labels']):
      if var > thresh:
        thresholded['variance'].append(var)
        thresholded['feature'].append(feat)

    vt_df = pd.DataFrame(thresholded)
    vt_df.to_csv(''.join(['./feature_selection/',file_name,'_vt','.csv']))
