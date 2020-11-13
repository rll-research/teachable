import pickle as pkl
import numpy as np
import torch
import blosc
import pathlib
from babyai.levels.iclr19_levels import *
from babyai.rl.utils.dictlist import DictList


def load(file_name):
    with open(file_name, 'rb') as f:
        loaded = pkl.load(f)
    return loaded


def save(file_name, file):
    with open(file_name, 'wb') as f:
        pkl.dump(file, f)


def theirs_to_ours(batch):
    traj_list = []
    for traj in batch:
        env = Level_PutNextLocalS5N3()

        # Obs
        obs = blosc.unpack_array(traj[1])
        instr = np.array(env.to_vocab_index(traj[0], pad_length=10))

        pre_action_advice = np.zeros((len(traj[3]), 8))
        indices = np.array(traj[3])
        pre_action_advice[np.arange(len(indices)), indices] = 1
        observations = []
        for o, p in zip(obs, pre_action_advice):
            obs_dict = {
                'obs': o,
                'instr': instr,
                'PreActionAdvice': p,
            }
            observations.append(obs_dict)

        # Action, teacher_action
        action = torch.IntTensor(traj[3])
        teacher_action = np.expand_dims(np.array(traj[3]), 1)
        # Full_done
        full_done = torch.zeros_like(action)
        full_done[-1] = 1
        our_dict = {
        'obs': observations,
        'action': action,
        'teacher_action': teacher_action,
        'full_done': full_done,
        }
        traj_list.append(DictList(our_dict))
    return traj_list

def ours_to_theirs(batch):
    env = Level_PutNextLocalS5N3()
    actions = batch.teacher_action[:, 0].tolist()

    # We don't have directions, so add a placeholder
    directions = np.zeros_like(actions).tolist()

    # obs
    obs = np.stack([d['obs'] for d in batch.obs])
    obs = blosc.pack_array(obs)

    # instr
    tokens = batch.obs[0]['instr']
    vocab = env.vocab()
    words = [vocab[token] for token in tokens if not token == 0]
    instr = ' '.join(words)

    their_tuple = (instr, obs, directions, actions)
    return their_tuple


def transform_all_theirs_to_ours(their_file, our_directory):
    theirs_train = load(their_file)
    print("loaded their data - train")
    ours_list_train = theirs_to_ours(theirs_train)
    print("transformed into our format - train")
    their_file_val = their_file[:-4] + '_valid.pkl'
    theirs_val = load(their_file_val)
    print("loaded their data - val")
    ours_list_val = theirs_to_ours(theirs_val)
    print("transformed into our format - val")

    our_directory = pathlib.Path(our_directory)
    if not our_directory.exists():
        our_directory.mkdir()
    for i, traj in enumerate(ours_list_train):
        file_name = our_directory.joinpath(f'traj_train_level18_idx{i}.pkl')
        save(file_name, traj)
    print("saved all train")
    for i, traj in enumerate(ours_list_val):
        file_name = our_directory.joinpath(f'traj_val_level18_idx{i}.pkl')
        save(file_name, traj)
    print("saved all val")


def transform_all_ours_to_theirs(our_directory, their_file):
    our_directory = pathlib.Path(our_directory)
    train_list = []
    val_list = []
    i = 0
    for file_name in our_directory.iterdir():
        if 'train' in file_name.name:
            train_list.append(load(file_name))
        else:
            val_list.append(load(file_name))
        i += 1
        if i % 10 == 0:
            print("loaded", i)
    print("loaded all files")
    theirs_train = [ours_to_theirs(traj) for traj in train_list]
    print("transformed all train")
    theirs_val = [ours_to_theirs(traj) for traj in val_list]
    print("transformed all val")
    save(their_file, theirs_train)
    print("saved train")
    their_file_val = their_file[:-4] + '_valid.pkl'
    save(their_file_val, theirs_val)
    print("saved val")