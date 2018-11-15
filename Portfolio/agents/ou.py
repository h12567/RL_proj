"""
Ornstein-Uhlenbeck Action Noise Algorithm
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""

import numpy as np


class OUAgent:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0

    def predict(self, s, a):
        s = s[0]
        a = a[0]

        x = get_close_price(s)
        x += self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        return x

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    @staticmethod
    def get_close_price(s):
        return [prices[-1][0] for i, prices in enumerate(s)]
