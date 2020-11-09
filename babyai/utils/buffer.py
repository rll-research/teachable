import random

import numpy as np
import pathlib
import pickle as pkl
import torch

from babyai.rl.utils.dictlist import merge_dictlists

# TODO: Currently we assume each batch comes from a single level. WE may need to change that assumption someday.
class Buffer:
    def __init__(self, path, buffer_capacity, prob_current):
        self.buffer_capacity = buffer_capacity
        # Probability that we sample from the current level instead of a past level
        self.prob_current = prob_current
        self.index = {}
        self.counts = {}
        self.buffer_path = pathlib.Path(path).joinpath('buffer')
        self.buffer_path.mkdir()

    def to_numpy(self, t):
        return t.detach().cpu().numpy()

    def split_batch(self, batch):
        # The batch is a series of trajectories concatenated. Here, we split it into individual batches.
        trajs = []
        end_idxs = self.to_numpy(torch.where(batch.full_done == 1)[0]) + 1
        start_idxs = np.concatenate([[0], end_idxs[:-1]])
        for start, end in zip(start_idxs, end_idxs):
            trajs.append(batch[start: end])
        return trajs

    def save_traj(self, traj, level, index):
        file_name = self.buffer_path.joinpath(f'traj_level{level}_idx{index}.pkl')
        with open(file_name, 'wb') as f:
            pkl.dump(traj, f)

    def load_traj(self, level, index):
        file_name = self.buffer_path.joinpath(f'traj_level{level}_idx{index}.pkl')
        with open(file_name, 'rb') as f:
            batch = pkl.load(f)
        return batch

    def add_trajs(self, batch, level):
        trajs = self.split_batch(batch)
        for traj in trajs:
            self.save_traj(traj, level, self.index[level])
            self.index[level] = (self.index[level] + 1) % self.buffer_capacity
            self.counts[level] = min(self.buffer_capacity, self.counts[level] + 1)

    def add_batch(self, batch, level):
        # Starting a new level
        if not level in self.index:
            self.counts[level] = 0
            self.index[level] = 0
        self.add_trajs(batch, level)

    def sample(self, total_num_samples):
        trajs = []
        num_samples = 0
        possible_levels = list(self.index.keys())
        while num_samples < total_num_samples:
            # With prob_current probability, sample from the latest level.
            if random.random() < self.prob_current:
                level = max(possible_levels)
            else:  # Otherwise, sample uniformly from the other levels
                level = random.choice(possible_levels)
            index = random.randint(0, self.counts[level] - 1)
            traj = self.load_traj(level, index)
            num_samples += len(traj.action)
            trajs.append(traj)
        # Combine our list of trajs in to a single DictList
        batch = merge_dictlists(trajs)
        return batch