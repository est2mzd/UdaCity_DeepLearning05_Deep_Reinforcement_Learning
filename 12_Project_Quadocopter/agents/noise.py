import numpy as np

class Ornstein_Uhlenbeck_Process:
    def __init__(self, size, mu, theta, sigma):
        self.size  = size
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
        np.random.seed(0)

    def reset(self):
        self.state = np.ones(self.size) * self.mu        

    def calc_noise(self):
        tmp_val = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state = self.state + tmp_val
        return self.state