import random

import numpy as np
import pathlib
import pickle as pkl
import torch

from babyai.rl.utils.dictlist import merge_dictlists, DictList

def trim_batch(batch):
    # Remove keys which aren't useful for distillation
    batch_info = {
        "obs": batch.obs[:-1],
        "next_obs": batch.obs[1:],
        "action": batch.action[:-1],
        "full_done": batch.full_done.int()[:-1],
        "success": batch.env_infos.success[:-1],
    }
    if 'action_probs' in batch:
        batch_info['action_probs'] = batch.action_probs[:-1]
    if 'teacher_action' in batch.env_infos:
        batch_info['teacher_action'] = batch.env_infos.teacher_action[:-1]
    if 'followed_teacher' in batch.env_infos:
        batch_info['followed_teacher'] = batch.env_infos.followed_teacher[:-1]
    if 'advice_count' in batch:
        batch_info['advice_count'] = batch.env_infos.advice_count[:-1]
    if 'reward' in batch:
        batch_info['reward'] = batch.reward[:-1]

    return DictList(batch_info)


# TODO: Currently we assume each batch comes from a single level. WE may need to change that assumption someday.
class Buffer:
    def __init__(self, path, buffer_capacity, prob_current, val_prob, buffer_name='buffer', augmenter=None,
                 successful_only=False):
        self.train_buffer_capacity = buffer_capacity
        self.augmenter = augmenter
        # We don't need that many val samples
        self.val_buffer_capacity = max(1, int(buffer_capacity * val_prob))
        # Probability that we sample from the current level instead of a past level
        self.prob_current = prob_current
        self.index_train = {}
        self.counts_train = {}
        self.index_val = {}
        self.counts_val = {}
        self.trajs_train = {}
        self.trajs_val = {}
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
        train_path = self.buffer_path.joinpath(f'train_buffer.pkl')
        if train_path.exists():
            with open(train_path, 'rb') as f:
                self.trajs_train, self.index_train, self.counts_train = pkl.load(f)
        val_path = self.buffer_path.joinpath(f'val_buffer.pkl')
        if val_path.exists():
            with open(self.buffer_path.joinpath(f'val_buffer.pkl'), 'rb') as f:
                self.trajs_val, self.index_val, self.counts_val = pkl.load(f)
        # if buffers are too big, trim them
        for level, count in self.counts_train.items():
            self.counts_train[level] = min(count, self.train_buffer_capacity)
        for level, index in self.index_train.items():
            self.index_train[level] = min(index, self.train_buffer_capacity - 1)
        for level, arr in self.trajs_train.items():
            if self.train_buffer_capacity < len(arr):
                self.trajs_train[level] = arr[:self.train_buffer_capacity]

    def create_blank_buffer(self, batch, label):
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
        self.trajs_train[label] = DictList(train_dict)
        self.trajs_val[label] = DictList(val_dict)

        pass

    def to_numpy(self, t):
        return t.detach().cpu().numpy()

    def split_batch(self, batch):
        # The batch is a series of trajectories concatenated. Here, we split it into individual batches.
        trajs = []
        end_idxs = self.to_numpy(torch.where(batch.full_done == 1)[0]) + 1
        start_idxs = np.concatenate([[0], end_idxs[:-1]])
        for start, end in zip(start_idxs, end_idxs):
            traj = batch[start: end]
            if (not self.successful_only) or traj.success[-1].item():
                trajs.append(traj)
                self.added_count += 1
        self.total_count += 1
        print("Buffer Counts", self.added_count, self.total_count, self.added_count/self.total_count)
        return trajs

    def save_traj(self, traj, level, index, split):
        if split == 'train':
            value = self.trajs_train[level]
        elif split == 'val':
            value = self.trajs_val[level]
        max_val = min(len(traj), len(value) - index)
        # We can fit the entire traj in
        for k in value:
            arr = getattr(value, k)
            try:
                arr[index:index + max_val] = getattr(traj, k)[:max_val]
            except Exception as e:
                print("???")

        # Uh oh, overfilling the buffer. Let's wrap around.
        remainder = min(len(traj) - max_val, len(arr))
        if remainder > 0:
            for k in value:
                arr = getattr(value, k)
                try:
                    arr[:remainder] = getattr(traj, k)[-remainder:]
                except:
                    print("???")

    def save_buffer(self):
        with open(self.buffer_path.joinpath(f'train_buffer.pkl'), 'wb') as f:
            pkl.dump((self.trajs_train, self.index_train, self.counts_train), f)
        with open(self.buffer_path.joinpath(f'val_buffer.pkl'), 'wb') as f:
            pkl.dump((self.trajs_val, self.index_val, self.counts_val), f)

    def add_trajs(self, batch, level, trim=True, only_val=False):
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
            self.save_traj(traj, level, self.index_val[level], 'val')
            self.index_val[level] = (self.index_val[level] + len(traj)) % self.val_buffer_capacity
            self.counts_val[level] = min(self.val_buffer_capacity, self.counts_val[level] + len(traj))
        for traj in trajs[split:]:
            self.save_traj(traj, level, self.index_train[level], 'train')
            self.index_train[level] = (self.index_train[level] + len(traj)) % self.train_buffer_capacity
            self.counts_train[level] = min(self.train_buffer_capacity, self.counts_train[level] + len(traj))
        self.save_buffer()
        print("COUNTS", self.counts_train[level], self.counts_val[level], self.index_train[level], self.index_val[level])

    def add_batch(self, batch, level, trim=True, only_val=False):
        # Starting a new level
        if not level in self.index_train:
            self.index_train[level] = 0
            self.index_val[level] = 0
            self.counts_val[level] = 0
            self.counts_train[level] = 0
            self.create_blank_buffer(trim_batch(batch), level)
        self.add_trajs(batch, level, trim, only_val)
        self.update_stats(batch)

    def update_stats(self, batch):
        for k in batch.obs[0].keys():
            if 'gave' in k:
                self.num_feedback += np.sum([o[k] for o in batch.obs])
        buffer_stats = self.counts_train, self.index_train, self.counts_val, self.index_val, self.num_feedback
        with open(self.buffer_path.joinpath('buffer_stats.pkl'), 'wb') as f:
            pkl.dump(buffer_stats, f)


    def sample(self, total_num_samples=None, split='train'):
        if split == 'train':
            index = self.index_train
            counts = self.counts_train
            trajs = self.trajs_train
        else:
            index = self.index_val
            counts = self.counts_val
            trajs = self.trajs_val
            # Early in training we may not have any val trajs yet
            if sum(counts.values()) == 0:
                index = self.index_train
                counts = self.counts_train
                trajs = self.trajs_train
        # Half from the latest level, otherwise choose uniformly from other levels  # TODO: later!!!
        possible_levels = list(index.keys())
        level = np.random.choice(possible_levels)
        indices = np.random.randint(0, counts[level], size=total_num_samples)
        data = merge_dictlists([trajs[level][i:i+1] for i in indices])
        return data
