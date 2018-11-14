"""
Follow-The-Winner Strategy
Always buy the stock that performs the best today
"""

import numpy as np


class Winner:
    def predict(self, s, a):
        returns = []
        for i, prices in enumerate(s[0]):
            cur_price = prices[-1][0]
            yst_price = prices[-2][0]
            returns.append(cur_price / yst_price)
        weights = np.zeros(len(s[0]))
        weights[np.argmax(returns)] = 1
        weights = weights[None, :]
        return weights
