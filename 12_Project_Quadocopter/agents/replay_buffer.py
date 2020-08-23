from collections import namedtuple, deque
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer      = deque(maxlen=buffer_size)
        self.experience  = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(1)
        
    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        if len(self.buffer) >= self.buffer_size: 
            self.buffer.popleft()
        self.buffer.append(experience)
        
    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size, action_size, state_size):
        experience_list = []
        
        if len(self.buffer) < batch_size:
            experience_list = random.sample(self.buffer, len(self.buffer))
        else:
            experience_list = random.sample(self.buffer, batch_size)
            
        state_list  = np.vstack([experience.state  for experience in experience_list if experience is not None])
        action_list = np.array( [experience.action for experience in experience_list if experience is not None]).astype(np.float32).reshape(-1, action_size)
        reward_list = np.array( [experience.reward for experience in experience_list if experience is not None]).astype(np.float32).reshape(-1, 1)
        done_list   = np.array( [experience.done   for experience in experience_list if experience is not None]).reshape(-1, 1)
        next_state_list = np.vstack([experience.next_state for experience in experience_list if experience is not None])
        
        return state_list, action_list, reward_list, done_list, next_state_list