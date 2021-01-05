import numpy as np
import copy
from gym_minigrid.minigrid import COLOR_NAMES, COLOR_TO_IDX, IDX_TO_COLOR, OBJECT_TO_IDX, IDX_TO_OBJECT
from babyai.rl.utils.dictlist import DictList


class DataAugmenter:

    def __init__(self, vocab):
        self.vocab = vocab

    def modify_obs(self, original_obs, valid_indices, permutation, check_index, modify_index):
        new_obs = original_obs.copy()
        for row in range(7):
            for col in range(7):
                if original_obs[row, col, check_index] in valid_indices:
                    target_color_idx = permutation[original_obs[row, col, modify_index]]
                    new_obs[row, col, modify_index] = target_color_idx
        return new_obs

    def modify_instr(self, instr, permutation, valid_words, word_to_idx, idx_to_word):
        modified_instr = []
        for our_instr_idx in instr:
            instr_word = self.vocab[our_instr_idx]
            if instr_word in valid_words:
                try:
                    target_color_idx = permutation[word_to_idx[instr_word]]
                except:
                    print("probs")
                target_color_name = idx_to_word[target_color_idx]
                our_target_color_idx = self.vocab.index(target_color_name)
                modified_instr.append(our_target_color_idx)
            else:
                modified_instr.append(our_instr_idx)
        modified_instr = np.array(modified_instr)
        return np.array(modified_instr)

    def make_permutation(self, valid_indices):
        valid_indices = np.array(valid_indices)
        permuted_indices = valid_indices.copy()
        np.random.shuffle(permuted_indices)
        permutation = {source: target for source, target in zip(valid_indices, permuted_indices)}
        return permutation

    def augment(self, original_traj, num_augmentations=1,
                augmentation_types=('color_permutation', 'object_permutation'), include_original=True):
        augmented_trajs = [copy.deepcopy(original_traj)] if include_original else []
        for _ in range(num_augmentations):
            traj = copy.deepcopy(original_traj)
            for augmentation in augmentation_types:
                if augmentation == 'color_permutation':
                    # Change Instruction
                    permutation = self.make_permutation(np.arange(len(COLOR_NAMES)))
                    modified_instr = self.modify_instr(traj.obs[0]['instr'], permutation, COLOR_NAMES, COLOR_TO_IDX,
                                                       IDX_TO_COLOR)
                    for obs_dict in traj.obs:
                        obs_dict['instr'] = modified_instr
                        # Change observation
                        # np.arange(4, 8) is the range of objects we're modifying: all the objs you can pick up + doors
                        obs_dict['obs'] = self.modify_obs(obs_dict['obs'], np.arange(4, 8), permutation, 0, 1)
                        # Change Feedback
                        if 'CartesianCorrections' in obs_dict:
                            original_cc3 = obs_dict['CartesianCorrections'].copy().reshape((7, 7, 3))
                            new_cc3 = self.modify_obs(original_cc3, np.arange(4, 8), permutation, 0, 1)
                            obs_dict['CartesianCorrections'] = new_cc3
                if augmentation == 'object_permutation':
                    permutation = self.make_permutation(np.arange(5, 8))  # objs you can pick up
                    modified_instr = self.modify_instr(traj.obs[0]['instr'], permutation,
                                                       [IDX_TO_OBJECT[k] for k in np.arange(5, 8)], OBJECT_TO_IDX,
                                                        IDX_TO_OBJECT)
                    for obs_dict in traj.obs:
                        obs_dict['instr'] = modified_instr
                        # Change observation
                        obs_dict['obs'] = self.modify_obs(obs_dict['obs'], np.arange(5, 8), permutation, 0, 0)
                        # Change Feedback
                        if 'CartesianCorrections' in obs_dict:
                            original_cc3 = obs_dict['CartesianCorrections'].copy().reshape((7, 7, 3))
                            new_cc3 = self.modify_obs(original_cc3, np.arange(5, 8), permutation, 0, 0)
                            obs_dict['CartesianCorrections'] = new_cc3
            augmented_trajs.append(traj)
            return augmented_trajs
