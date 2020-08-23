import numpy as np
import gym
import matplotlib.pyplot as plt


class Actor():
  def __init__(self, env):
    num_row = env.observation_space.n
    num_col = env.action_space.n
    self.actions = list(range(num_col))
    self.Q = np.random.uniform(0,1,num_row*num_col).reshape((num_row, num_col))

  # ------------------------------------------------------------
  def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

  # ------------------------------------------------------------
  def policy(self, state):
    tmp_action = np.random.choice(self.actions, 1, p=self.softmax(self.Q[state]))
    return tmp_action[0]



class Critic():
  def __init__(self, env):
    states = env.observation_space.n
    # self.V = np.zeros(states)
    self.V = np.random.uniform(0,1,states)#.reshape((1, states))