import random

import numpy as np
import pathlib
import pickle as pkl
import torch

from utils.dictlist import merge_dictlists, DictList


def trim_batch(batch):
    # Remove keys which aren't useful for distillation
    batch_info = {
        "obs": batch.obs,
        "next_obs": batch.next_obs,
        "action": batch.action,
        "full_done": batch.full_done,
        "success": batch.env_infos.success,
    }
    if 'action_probs' in batch:
        batch_info['action_probs'] = batch.action_probs
    if 'teacher_action' in batch.env_infos:
        batch_info['teacher_action'] = batch.teacher_action
    if 'followed_teacher' in batch.env_infos:
        batch_info['followed_teacher'] = batch.env_infos.followed_teacher
    if 'advice_count' in batch:
        batch_info['advice_count'] = batch.env_infos.advice_count
    if 'reward' in batch:
        batch_info['reward'] = batch.reward

    return DictList(batch_info)


class Buffer:
    def __init__(self, path, buffer_capacity, val_prob, buffer_name='buffer', successful_only=False):
        self.train_buffer_capacity = buffer_capacity
        # We don't need that many val samples
        self.val_buffer_capacity = max(1, int(buffer_capacity * val_prob))
        self.counts_train, self.index_train, self.counts_val, self.index_val = 0, 0, 0, 0
        self.trajs_train, self.trajs_val = None, None
        self.buffer_path = pathlib.Path(path).joinpath(buffer_name)
        self.successful_only = successful_only
        self.num_feedback = 0
        # If the buffer already exists, load it
        if self.buffer_path.exists():
            self.load_buffer()
        else:
            self.buffer_path.mkdir()
        self.val_prob = val_prob
        self.added_count = 0
        self.total_count = 0

    def load_buffer(self):
        """ Load buffer from pkl file. """
        train_path = self.buffer_path.joinpath(f'train_buffer.pkl')
        if train_path.exists():
            with open(train_path, 'rb') as f:
                self.trajs_train, self.index_train, self.counts_train = pkl.load(f)
        val_path = self.buffer_path.joinpath(f'val_buffer.pkl')
        if val_path.exists():
            with open(self.buffer_path.joinpath(f'val_buffer.pkl'), 'rb') as f:
                self.trajs_val, self.index_val, self.counts_val = pkl.load(f)
        # if buffers are too big, trim them
        self.counts_train = min(self.counts_train, self.train_buffer_capacity)
        self.index_train = min(self.index_train, self.train_buffer_capacity - 1)
        self.trajs_train = self.trajs_train[:self.train_buffer_capacity]

    def create_blank_buffer(self, batch):
        """ Create blank buffer with all keys. (We don't do this at startup b/c we don't know all the batch keys.) """
        batch = trim_batch(batch)
        train_dict = {}
        val_dict = {}
        for key in list(batch.keys()):
            try:
                value = getattr(batch, key)
                if type(value) is list:
                    train_dict[key] = [None] * self.train_buffer_capacity
                    val_dict[key] = [None] * self.val_buffer_capacity
                elif type(value) is torch.Tensor:
                    shape = value.shape
                    tensor_class = torch.IntTensor if value.dtype is torch.int32 else torch.FloatTensor
                    device = value.device
                    train_dict[key] = tensor_class(size=(self.train_buffer_capacity, *shape[1:])).to(device)
                    val_dict[key] = tensor_class(size=(self.val_buffer_capacity, *shape[1:])).to(device)
                elif type(value) is np.ndarray:
                    shape = value.shape
                    dtype = value.dtype
                    train_dict[key] = np.zeros(shape=(self.train_buffer_capacity, *shape[1:]), dtype=dtype)
                    val_dict[key] = np.zeros(shape=(self.val_buffer_capacity, *shape[1:]), dtype=dtype)
                else:
                    raise NotImplementedError((key, type(value)))
            except:
                print("?", key)
        self.trajs_train = DictList(train_dict)
        self.trajs_val = DictList(val_dict)

        pass

    def to_numpy(self, t):
        """ Torch tensor -> numpy array """
        return t.detach().cpu().numpy()

    def split_batch(self, batch):
        """ The batch is a series of trajectories concatenated. Here, we split it into individual batches. """
        trajs = []
        end_idxs = self.to_numpy(torch.where(batch.full_done == 1)[0]) + 1
        start_idxs = np.concatenate([[0], end_idxs[:-1]])
        for start, end in zip(start_idxs, end_idxs):
            traj = batch[start: end]
            if (not self.successful_only) or traj.success[-1].item():
                trajs.append(traj)
                self.added_count += 1
        self.total_count += 1
        print("Buffer Counts", self.added_count, self.total_count, self.added_count / self.total_count)
        return trajs

    def save_traj(self, traj, index, split):
        """ Insert a trajectory into the buffer """
        value = self.trajs_train if split == 'train' else self.trajs_val
        max_val = min(len(traj), len(value) - index)
        # We can fit the entire traj in
        for k in value:
            arr = getattr(value, k)
            arr[index:index + max_val] = getattr(traj, k)[:max_val]
        # Uh oh, overfilling the buffer. Let's wrap around.
        remainder = min(len(traj) - max_val, len(arr))
        if remainder > 0:
            for k in value:
                getattr(value, k)[:remainder] = getattr(traj, k)[-remainder:]

    def save_buffer(self):
        """ Save buffer to pkl files. """
        with open(self.buffer_path.joinpath(f'train_buffer.pkl'), 'wb') as f:
            pkl.dump((self.trajs_train, self.index_train, self.counts_train), f)
        with open(self.buffer_path.joinpath(f'val_buffer.pkl'), 'wb') as f:
            pkl.dump((self.trajs_val, self.index_val, self.counts_val), f)

    def add_trajs(self, batch, trim=True, only_val=False):
        """ Save a batch of trajectories, passed in as a Dictlist of timesteps of sequential trajs. """
        if trim:
            batch = trim_batch(batch)
        trajs = self.split_batch(batch)
        random.shuffle(trajs)
        split = int(self.val_prob * len(trajs))
        # Make sure we get at least one of each
        if split == 0 and len(trajs) > 1:
            split = 1
        if only_val:
            split = len(trajs)
        for traj in trajs[:split]:
            self.save_traj(traj, self.index_val, 'val')
            self.index_val = (self.index_val + len(traj)) % self.val_buffer_capacity
            self.counts_val = min(self.val_buffer_capacity, self.counts_val + len(traj))
        for traj in trajs[split:]:
            self.save_traj(traj, self.index_train, 'train')
            self.index_train = (self.index_train + len(traj)) % self.train_buffer_capacity
            self.counts_train = min(self.train_buffer_capacity, self.counts_train + len(traj))
        self.save_buffer()
        print("COUNTS", self.counts_train, self.counts_val, self.index_train, self.index_val)

    def add_batch(self, batch, trim=True, only_val=False):
        """ Save a batch of data and update counters. Data is a Dictlist of timesteps of sequential trajs.
         This is the function which is called externally. """
        if self.trajs_train is None:
            self.create_blank_buffer(batch)
        self.add_trajs(batch, trim, only_val)
        self.update_stats(batch)

    def update_stats(self, batch):
        """ Save pointers to our current index in the buffer and some counts. """
        for k in batch.obs[0].keys():
            if 'gave' in k:
                self.num_feedback += np.sum([o[k] for o in batch.obs])
        buffer_stats = self.counts_train, self.index_train, self.counts_val, self.index_val, self.num_feedback
        with open(self.buffer_path.joinpath('buffer_stats.pkl'), 'wb') as f:
            pkl.dump(buffer_stats, f)

    def sample(self, total_num_samples=None, split='train'):
        """ Sample a batch. """
        if split == 'train' or self.counts_val == 0:  # Early in training we may not have any val trajs yet
            counts = self.counts_train
            trajs = self.trajs_train
        else:
            counts = self.counts_val
            trajs = self.trajs_val

        indices = np.random.randint(0, counts, size=total_num_samples)
        data = merge_dictlists([trajs[i:i + 1] for i in indices])
        return data
