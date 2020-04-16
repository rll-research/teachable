from meta_mb.optimizers.base import Optimizer
from collections import OrderedDict
import numpy as np


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


class RandomSearchOptimizer(Optimizer):
    def __init__(self, policy, num_deltas, learning_rate, percentile=0.9):
        self.num_deltas = num_deltas
        self.percentile = percentile
        self.policy = policy
        self.learning_rate = learning_rate

        self.policy_keys = self.policy.policy_params.keys()

    def _get_deltas(self, deltas, idxs):
        deltas_idxs = OrderedDict()
        for k in self.policy_keys:
            deltas_idxs[k] = deltas[k][idxs, ...]
        return deltas_idxs

    def _compute_gradient(self, scale, direction):
        # TODO: In the original ARS implementation they compute the gradient with the batched_weighted_sum -- I dunno why
        g_hat = OrderedDict()
        for k in self.policy_keys:
            g_hat[k] = np.mean(scale * direction[k].T, axis=-1).T
        return g_hat

    def _apply_gradient(self, g_hat):
        for k, v in self.policy.policy_params.items():
            self.policy.policy_params[k] = v + self.learning_rate * g_hat[k]

    def optimize_policy(self, returns, deltas):
        assert returns.shape == (self.num_deltas*2,)
        assert all([len(returns) == len(v) * 2 for v in deltas.values()])
        returns = np.array(np.split(returns, 2)).T
        max_returns = np.max(returns, axis=1)

        # Compute the top performing directions
        idx = np.arange(self.num_deltas)[max_returns >= np.percentile(returns, self.percentile)]

        returns = returns[idx, :]
        returns /= np.std(returns)

        deltas_idxs = self._get_deltas(deltas, idx)
        g_hat = self._compute_gradient(returns[:, 0] - returns[:, 1], deltas_idxs)

        # I separated both functions in case we want to do some logging
        self._apply_gradient(g_hat)



