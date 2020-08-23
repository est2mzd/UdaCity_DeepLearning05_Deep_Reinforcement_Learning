# 説明
# The interact function tests how well your agent learns from
# interaction with the environment.

from collections import deque
import sys
import math
import numpy as np
from Actor_Critic import *

def interact_AC(env, num_episodes=20000, window=100, gamma=0.9, learning_rate=0.1, momentum=0.99):
    """ Monitor agent's performance.
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)

    actor  = Actor(env)
    critic = Critic(env)

    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        while True:
            action = actor.policy(state)
            next_state, reward, done, info = env.step(action)
            gain      = reward + gamma * critic.V[next_state]
            estimated = critic.V[state]
            td_diff   = estimated - gain
            #
            actor.Q[state][action] = momentum * actor.Q[state][action] - learning_rate * td_diff
            critic.V[state]        = momentum * critic.V[state]        - learning_rate * td_diff
            #
            state = next_state
            samp_reward += reward

            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break

        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward