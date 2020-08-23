# 説明
# Run this ﬁle in the terminal to check the performance of your agent.
# Leaders Board
#   Taxi-v2 => [1]9.71, [2]9.70 ,[3]9.63
#   Taxi-v3 => [1]9.07, [2]8.80 ,[3]8.57
#
# 自分の環境では、Taxi-v2ができなかったので、Taxi-v3をトライした
#
# Actor-Critic　の　能力を試したかったので、monitor.pyを新規に作成した

from monitor_Actor_Critic import *
import gym
import numpy as np

# env = gym.make('Taxi-v2')
env = gym.make('Taxi-v3')

num_episodes = 20000
window=100
momentum = 1.000

# gamma_list = [1.0, 0.9, 0.8]
# lr_list    = [0.01,0.1,0.3,0.5]

# 初期値を、zerosからuniformに変えたら、収束速度がかなり向上した！
#  =>> 報酬の立ち上がりがかなり早くなった
gamma_list = [1.0]
lr_list    = [0.08,0.09,0.10,0.11,0.12]

for gamma in gamma_list:
  for learning_rate in lr_list:
    print("gamma = {} / learning_rate = {}".format(gamma, learning_rate))
    avg_rewards, best_avg_reward = interact_AC(env, num_episodes=num_episodes, window=window, gamma=gamma, learning_rate=learning_rate, momentum=momentum)

