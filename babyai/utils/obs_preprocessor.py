import torch
import numpy as np
from babyai.rl.utils.dictlist import DictList

def make_obs_preprocessor(teacher_null_dict, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    def obss_preprocessor(obs, teacher_dict):
        obs_output = {}

        # Populate dictionary with an empty dict
        for k in obs[0].keys():
            # Don't have individual elements for the advice, since we concat these together
            # We might consider changing this if we process diff advice types differently (e.g. cartesian with a conv net)
            if not k in teacher_dict:
                obs_output[k] = []
            obs_output['advice'] = []

        for o in obs:
            advice_list = []
            for k, v in o.items():
                if k in teacher_dict:
                    # Mask out particular teachers
                    if not teacher_dict[k]:
                        v = teacher_null_dict[k]
                    advice_list.append(v.flatten())
                else:
                    obs_output[k].append(v)
            if len(advice_list) > 0:
                obs_output['advice'].append(np.concatenate(advice_list))

        for k, v in obs_output.items():
            obs_output[k] = torch.FloatTensor(v).to(device)
        return DictList(obs_output)

    return obss_preprocessor