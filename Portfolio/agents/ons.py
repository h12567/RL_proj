"""
Online Newton Step Algorithm
http://www.cs.princeton.edu/~ehazan/papers/icml.pdf
"""

import numpy as np


class ONSAgent:
    def __init__(self, delta=0.125, beta=1., eta=0.):
        self.delta = delta
        self.beta = beta
        self.eta = eta

        self.A = None

    def predict(self, s, a):
        s = s[0]
        a = a[0]

        n = len(s)
        if a is None:
            self.A = np.mat(np.eye(n))
            weights = np.array([1. / n] * n)
            weights = weights[None, :]
        else:
            x = get_close_price(s)
            grad = np.mat(x / np.dot(a, x)).T

            self.A += grad * grad.T
            self.b += (1 + 1. / self.beta) * grad

            pp = projection_in_norm(self.delta * self.A.I * self.b, self.A)
            return pp * (1 - self.eta) + np.ones(len(x)) / float(len(x)) * self.eta

        return weights

    @staticmethod
    def get_close_price(s):
        return [prices[-1][0] for i, prices in enumerate(s)]

    @staticmethod
    def projection_in_norm(x, mat):
        """ Projection of x to simplex indiced by matrix M. Uses quadratic programming.
        """
        m = mat.shape[0]

        P = matrix( 2 * mat)
        q = matrix(-2 * mat * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m,1)))
        A = matrix(np.ones((1,m)))
        b = matrix(1.)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol['x'])
