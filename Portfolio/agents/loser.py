"""
Follow-The-Losser Strategy
Always buy the stock that performs the worst today
"""

import numpy as np


class Loser:
    def predict(self, s, a):
        s = s[0]
        returns = [prices[-1][0] / prices[-2][0] for i, prices in enumerate(s)]

        weights = np.zeros(len(s))
        weights[np.argmin(returns)] = 1
        weights = weights[None, :]
        return weights
