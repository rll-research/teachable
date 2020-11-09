import random

import numpy as np
import torch

from babyai.rl.utils.dictlist import merge_dictlists

# TODO: Currently we assume each batch comes from a single level. WE may need to change that assumption someday.
class Buffer:
    def __init__(self, buffer_capacity, prob_current):
        self.buffer_capacity = buffer_capacity
        # Probability that we sample from the current level instead of a past level
        self.prob_current = prob_current
        self.buffer = []
        self.index = []

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

    def add_batch(self, batch, level):
        # Starting a new level
        if level >= len(self.buffer):
            self.buffer.append(self.split_batch(batch))
            self.index.append(1)
        else:
            # Existing level, buffer isn't full yet
            level_buffer = self.buffer[level]
            if len(level_buffer) < self.buffer_capacity:
                trajs = self.split_batch(batch)
                # If we will exceed the buffer capacity midway through this trajectory...
                if len(level_buffer) + len(trajs) > self.buffer_capacity:
                    spaces_free = self.buffer_capacity - len(level_buffer)
                    self.add_to_unfull_buffer(trajs[:spaces_free], level)
                    self.add_to_full_buffer(trajs[spaces_free:], level)
                else:
                    self.add_to_unfull_buffer(trajs, level)
            else:
                # Existing level, buffer is full
                self.add_to_full_buffer(self.split_batch(batch), level)


    def add_to_full_buffer(self, trajs, level):
        level_buffer = self.buffer[level]
        for traj in trajs:
            index = self.index[level]
            level_buffer[index] = traj
            self.index[level] = (len(trajs) + 1) % self.buffer_capacity

    def add_to_unfull_buffer(self, trajs, level):
        level_buffer = self.buffer[level]
        level_buffer += trajs
        self.index[level] = (len(trajs) + len(trajs)) % self.buffer_capacity

    def sample(self, total_num_samples):
        trajs = []
        num_samples = 0
        while num_samples < total_num_samples:
            # With prob_current probability, sample from the latest level.
            if random.random() < self.prob_current:
                level_buffer = self.buffer[-1]
            else:  # Otherwise, sample uniformly from the other levels
                level_buffer = random.choice(self.buffer)
            traj = random.choice(level_buffer)
            num_samples += len(traj.action)
            trajs.append(traj)
        # Combine our list of trajs in to a single DictList
        batch = merge_dictlists(trajs)
        return batch