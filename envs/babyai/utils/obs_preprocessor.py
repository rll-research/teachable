import torch
import numpy as np
from utils.dictlist import DictList

def make_obs_preprocessor(feedback_list, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          pad_size=51):
    def obss_preprocessor(obs, teacher, show_instrs=True):
        obs_output = {}
        assert not 'advice' in obs[0].keys(), "Appears to already be preprocessed"

        instr_mask = int(show_instrs)
        obs_output['advice'] = [o[teacher] for o in obs if teacher in o]
        if 'instr' in obs[0]:
            obs_output['instr'] = [o['instr'] for o in obs]
        obs_output['obs'] = [o['obs'] for o in obs]

        obs_final = {}
        for k, v in obs_output.items():
            if k == 'obs' and type(v[0]) is tuple:  # Padding for egocentric view
                obs_final[k] = np.zeros((len(v), pad_size, pad_size, 3))
                middle = int(pad_size / 2)
                for i, (img, x, y) in enumerate(v):
                    y_start = middle - y
                    x_start = middle - x
                    obs_final[k][i][x_start:x_start + len(img), y_start:y_start + len(img[0])] = img
                obs_final[k] = torch.FloatTensor(obs_final[k]).to(device)
            else:
                obs_final[k] = torch.FloatTensor(v).to(device)
            if k == 'instr':
                obs_final[k] = obs_final[k] * instr_mask
        return DictList(obs_final)

    return obss_preprocessor


def obss_preprocessor_distill(obs):
    obs_output = {k: [] for k in obs[0].keys()}
    for o in obs:
        for k, v in o.items():
            obs_output[k].append(v)
    obs_final = {}
    for k, v in obs_output.items():
        obs_final[k] = np.stack(v)
    return DictList(obs_final)


def make_obs_preprocessor_choose_teachers(teacher_null_dict, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          include_zeros=True):
    def obss_preprocessor_choose_teachers(obs, teacher_dict, show_instrs=True):
        obs_output = {}
        assert not 'advice' in obs[0].keys(), "Appears to already be preprocessed"
        obs_output['obs'] = torch.FloatTensor(obs.obs).to(device)
        obs_output['instr'] = torch.FloatTensor(obs.instr * int(show_instrs)).to(device)
        teacher_data = []
        for k, v in teacher_dict.items():
            if teacher_dict[k]:
                teacher_data.append(getattr(obs, k))
            elif include_zeros:
                teacher_data.append(np.concatenate([teacher_null_dict[k]] * len(obs.obs)))
        if len(teacher_data) > 0:
            obs_output['advice'] = torch.FloatTensor(np.concatenate(teacher_data, axis=1)).to(device)
        return DictList(obs_output)
    return obss_preprocessor_choose_teachers
