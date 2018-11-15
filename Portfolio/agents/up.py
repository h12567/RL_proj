"""
Universal Portfolio Algorithm
Professor Thomas M. Cover
"""

import numpy as np


class UPAgent:
    def __init__(self, eval_points=10000, leverage=1., W=None):
        self.eval_points = eval_points
        self.leverage = leverage
        self.W = W

        self.S = None

    def init_portfolio(self, close_price):
        m = close_price.shape[1]

        # create set of CRPs
        self.W = np.matrix(mc_simplex(m - 1, self.eval_points))
        self.S = np.matrix(np.ones(self.W.shape[0])).T

        # stretch simplex based on leverage (simple calculation yields this)
        leverage = max(self.leverage, 1. / m)
        stretch = (leverage - 1. / m) / (1. - 1. / m)
        self.W = (self.W - 1. / m) * stretch + 1. / m

    def predict(self, s, a):
        s = s[0]
        a = a[0]

        close_price = get_close_price(s)
        close_price = np.reshape(close_price, (1, close_price.size))

        if self.W is None:
            self.init_portfolio(close_price)

        self.S = np.multiply(self.S, self.W * np.matrix(close_price).T)
        a = self.W.T * self.S
        pv = a / np.sum(a)
        pvn = np.ravel(pv)

        return pvn

    @staticmethod
    def get_close_price(s):
        return [prices[-1][0] for i, prices in enumerate(s)]
