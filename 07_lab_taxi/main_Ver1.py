# 説明
# Run this ﬁle in the terminal to check the performance of your agent.
# Leaders Board
#   Taxi-v2 => [1]9.71, [2]9.70 ,[3]9.63
#   Taxi-v3 => [1]9.07, [2]8.80 ,[3]8.57
#
# 自分の環境では、Taxi-v2ができなかったので、Taxi-v3をトライした
# 下記の設定で、結果が 8.80　以上になるので、収束できていると判断した
# 
# epsilon は、常に一定値だと 高い報酬が得られなかった
# 初期は　[eps = 1/i_episode]　として徐々に大きくし
# eps_max　で、最大値を規定すると、高い報酬が得られるようになった

from agent import Agent
from monitor import interact
import gym
import numpy as np

# env = gym.make('Taxi-v2')
env = gym.make('Taxi-v3')

agent = Agent()
agent.eps_max     = 0.1
agent.gamma       = 1.0
agent.update_type = 1   # 1=Q-Learning / 2=Expected SARSA
print("esp = {} / gamma = {}".format(agent.eps_max, agent.gamma))

# avg_rewards, best_avg_reward = interact(env, agent, num_episodes=5000, window=100)
avg_rewards, best_avg_reward = interact(env, agent)

