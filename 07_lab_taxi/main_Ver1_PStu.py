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
#
# 下記の設定で、ハイパーパラメータの調整を実行した

from agent import Agent
from monitor import interact
import gym
import numpy as np

# env = gym.make('Taxi-v2')
env = gym.make('Taxi-v3')

#eps_list = [0.002,0.004,0.006,0.008,0.010,0.012,0.014,0.016,0.018]
# eps_list   = [0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18]
# eps_list   = [0.02,0.04,0.06]
# eps_list   = [0.08,0.10,0.12]
eps_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# gamma_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
gamma_list = [1.0,0.99,0.98,0.97,0.96]

res_avg = []
res_best_avg = []

for eps in eps_list:
  for gamma in gamma_list:
    agent = Agent()
    agent.eps_max     = eps
    agent.gamma       = gamma
    agent.update_type = 1     # 1=Q-Learning / 2=Expected SARSA
    print("esp = {} / gamma = {}".format(eps, gamma))
    # avg_rewards, best_avg_reward = interact(env, agent, num_episodes=5000, window=100)
    avg_rewards, best_avg_reward = interact(env, agent)
    #
    res_avg.append(avg_rewards)
    res_best_avg.append(best_avg_reward)

