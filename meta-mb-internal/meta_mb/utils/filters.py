# Code in this file is copied and adapted from
# https://github.com/ray-project/ray/blob/master/python/ray/rllib/utils/filter.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Filter(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x, *args, **kwargs):
        return x

    def stats_increment(self):
        pass

    def get_params(self):
        return None

    def set_params(self, *args, **kwargs):
        pass


class MeanStdFilter(Filter):
    def __init__(self, shape):
        self.shape = shape
        self._n = 0
        self._M = np.zeros(shape, dtype=np.float64)
        self._S = np.zeros(shape, dtype=np.float64)

        self.mean = np.zeros(shape, dtype=np.float64)
        self.std = np.ones(shape, dtype=np.float64)

    def __call__(self, x, update=True):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape[-len(self.shape):] == self.shape
        if update:
            self._update(x)
        x = x - self.mean
        x = x / (self.std + 1e-8)
        return x

    def stats_increment(self):
        self.mean = self._M.copy()
        self.std = np.sqrt(self._S).copy()

    def _update(self, x):
        x = np.reshape(x, (-1,) + self.shape)
        n1 = self._n
        n2 = len(x)
        n = n1 + n2
        mean = np.mean(x, axis=0)
        M = (n1 * self._M + n2 * mean)/n

        delta = self._M - mean
        delta2 = np.square(delta)
        var = np.var(x, axis=0)
        S = (self._S * n1 + var * n2 + n1 * n2/n * delta2)/n

        self._M = M
        self._n = n
        self._S = S

    def get_params(self):
        return self.mean, self.std

    def set_params(self, mean, std):
        self._M = mean
        self._S = np.square(std)
        self.stats_increment()


