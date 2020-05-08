import argparse
from experiment_utils.utils import load_exps_data
import os
import json

"""
 python /home/ignasi/GitRepos/meta-mb/experiment_utils/save_videos.py data/s3/mbmpo-pieter/ --speedup 4 -n 1 --max_path_length 300 --ignore_done
"""


def modify_json(json_file, exp_name):
    json_file['exp_tag'] = 'a-me-trpo'
    json_file['async'] = True
    json_file['algo'] = 'a-me-trpo'
    return json_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    args = parser.parse_args()

    experimet_paths = load_exps_data(args.data, gap=0.)
    counter = 0
    for exp_path in experimet_paths:
        json_file = exp_path['json']
        json_path = os.path.join(exp_path['exp_name'], 'params.json')
        new_json_file = modify_json(json_file, exp_path['exp_name'])
        json.dump(new_json_file, open(json_path, 'w'))


