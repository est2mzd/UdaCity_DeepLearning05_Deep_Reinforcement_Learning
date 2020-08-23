# 説明
# Develop your reinforcement learning agent here.
# This is the only ﬁle that you should modify.

from utility_for_pycharm import *

import numpy as np
from collections import defaultdict


class Agent:

  # ------------------------------------------------------------
  def __init__(self, nA=6):
    """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
    self.nA = nA
    self.Q = defaultdict(lambda: np.zeros(self.nA))
    self.eps     = 0.05
    self.eps_max = 0.15
    self.cnt_episode = 0
    #
    self.alpha = 0.01
    self.gamma = 1.0
    #
    self.update_type = 1

  # ------------------------------------------------------------
  def select_action(self, state,i_episode):
    # 入力 : state
    # 出力 : action
    """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
    """
    if (1 / i_episode) < self.eps_max:
      self.eps = 1 / i_episode
    else:
      self.eps = self.eps_max

    action = epsilon_greedy(self.Q, state, self.nA, self.eps)
    return action

  # ------------------------------------------------------------
  def step(self, state, action, reward, next_state, done):
    """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

    if self.update_type == 1:
      # Q_learning
      self.Q[state][action] = self.update_by_Q_Learning(self.alpha, self.gamma, self.Q, state, action, reward, next_state)
    elif self.update_type == 2:
      # Expecte SARSA
      self.Q[state][action] = self.update_by_Expected_SARSA(self.alpha, self.gamma, self.Q, state, action, reward, self.nA, self.eps, next_state)

  #------------------------------------------------------------
  def update_by_Q_Learning(self, alpha, gamma, Q, state, action, reward, next_state=None):
    # Q_learning
    Q_current_experience = Q[state][action]
    #
    if next_state is not None:
      Q_next = np.max(Q[next_state])
    else:
      Q_next = 0
    #
    Q_current_predict = reward + (gamma * Q_next)
    new_value         = Q_current_experience + (alpha *(Q_current_predict - Q_current_experience))
    return new_value

  #------------------------------------------------------------
  def update_by_Expected_SARSA(self, alpha, gamma, Q, state, action, reward, nA, eps, next_state=None):
    """Returns updated Q-value for the most recent experience."""
    current = Q[state][action]  # estimate in Q-table (for current state, action pair)
    policy_s = np.ones(nA) * eps / nA  # current policy (for next state S')
    policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA)  # greedy action
    Qsa_next = np.dot(Q[next_state], policy_s)  # get value of state at next time step
    target = reward + (gamma * Qsa_next)  # construct target
    new_value = current + (alpha * (target - current))  # get updated value
    return new_value

