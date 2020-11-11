import random

import numpy as np
import pathlib
import pickle as pkl
import torch

from babyai.rl.utils.dictlist import merge_dictlists, DictList


def trim_batch(batch):
    # Remove keys which aren't useful for distillation
    return DictList({
        "obs": batch.obs,
        "action": batch.action.int(),
        "teacher_action": batch.env_infos.teacher_action,
        "full_done": batch.full_done.int()
    })


# TODO: Currently we assume each batch comes from a single level. WE may need to change that assumption someday.
class Buffer:
    def __init__(self, path, buffer_capacity, prob_current, val_prob):
        self.train_buffer_capacity = buffer_capacity
        # We don't need that many val samples
        self.val_buffer_capacity = np.clip(int(buffer_capacity * val_prob), 1, 50)
        # Probability that we sample from the current level instead of a past level
        self.prob_current = prob_current
        self.index_train = {}
        self.counts_train = {}
        self.index_val = {}
        self.counts_val = {}
        self.buffer_path = pathlib.Path(path).joinpath('buffer')
        self.buffer_path.mkdir()
        self.val_prob = val_prob

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
        batch = trim_batch(batch)
        trajs = self.split_batch(batch)
        random.shuffle(trajs)
        split = int(self.val_prob * len(trajs))
        # Make sure we get at least one of each
        if split == 0 and len(trajs) > 1:
            split = 1
        for traj in trajs[:split]:
            self.save_traj(traj, level, self.index_val[level])
            self.index_val[level] = (self.index_val[level] + 1) % self.val_buffer_capacity
            self.counts_val[level] = min(self.val_buffer_capacity, self.counts_val[level] + 1)
        for traj in trajs[split:]:
            self.save_traj(traj, level, self.index_train[level])
            self.index_train[level] = (self.index_train[level] + 1) % self.train_buffer_capacity
            self.counts_train[level] = min(self.train_buffer_capacity, self.counts_train[level] + 1)

    def add_batch(self, batch, level):
        # Starting a new level
        if not level in self.index_train:
            self.counts_train[level] = 0
            self.index_train[level] = 0
            self.counts_val[level] = 0
            self.index_val[level] = 0
        self.add_trajs(batch, level)

    def sample(self, total_num_samples, split='train'):
        if split == 'train':
            index = self.index_train
            counts = self.counts_train
        else:
            index = self.index_val
            counts = self.counts_val

        trajs = []
        num_samples = 0
        possible_levels = list(index.keys())
        while num_samples < total_num_samples:
            # With prob_current probability, sample from the latest level.
            if random.random() < self.prob_current:
                level = max(possible_levels)
            else:  # Otherwise, sample uniformly from the other levels
                level = random.choice(possible_levels)
            index = random.randint(0, counts[level] - 1)
            traj = self.load_traj(level, index)
            num_samples += len(traj.action)
            trajs.append(traj)
        # Combine our list of trajs in to a single DictList
        batch = merge_dictlists(trajs)
        return batch
