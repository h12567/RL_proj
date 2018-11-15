"""
Follow-The-Winner Strategy
Always buy the stock that performs the best today
"""

import numpy as np


class Winner:
    def predict(self, s, a):
        s = s[0]
        returns = [prices[-1][0] / prices[-2][0] for i, prices in enumerate(s)]

        weights = np.zeros(len(s))
        weights[np.argmax(returns)] = 1
        weights = weights[None, :]
        return weights

