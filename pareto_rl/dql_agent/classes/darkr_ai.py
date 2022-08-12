import torch.nn as nn
from collections import namedtuple, deque
import random
import torch
class DarkrAI(nn.Module):

  def __init__(self, input_size, layers, output_size):
    r"""
    Args:
      input_size: number of inputs
      layers: ordered list of hidden layer lengths
      output_size: size last layer and number of outputs
    """
    super(DarkrAI, self).__init__()
    self.layers = nn.ModuleList()
    # input layer
    self.layers.append(nn.Linear(input_size, layers[0], dtype=torch.float64))
    # hidden layers
    for i in range(1, len(layers)):
      self.layers.append(nn.Linear(layers[i-1], layers[i], dtype=torch.float64))
    # output layer
    self.layers.append(nn.Linear(layers[-1], output_size, dtype=torch.float64))
    self.activation = nn.ReLU()

  def forward(self, turn_input):
    x = turn_input
    for layer in self.layers:
      x = self.activation(layer(x))
    return x

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
