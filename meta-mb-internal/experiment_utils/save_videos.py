import joblib
import json
import numpy as np
import tensorflow as tf
import argparse
from meta_mb.samplers.utils import rollout
from experiment_utils.utils import load_exps_data

"""
 python /home/ignasi/GitRepos/meta-mb/experiment_utils/save_videos.py data/s3/mbmpo-pieter/ --speedup 4 -n 1 --max_path_length 300 --ignore_done
"""


def valid_experiment(params):
    # values = {'max_path_length': [200],
    #           'dyanmics_hidden_nonlinearity': ['relu'],
    #           'dynamics_buffer_size': [10000],
    #           'env': [{'$class': 'meta_mb.envs.mujoco.walker2d_env.Walker2DEnv'}]}

    # 'env': [{'$class': 'meta_mb.envs.mujoco.walker2d_env.Walker2DEnv'}]}
    # 'env': [{'$class': 'meta_mb.envs.mujoco.ant_env.AntEnv'}]}
    # 'env': [{'$class': 'meta_mb.envs.mujoco.hopper_env.HopperEnv'}]}
    # #
    values = {'max_path_length': [200],
              'num_rollouts': [100],
              'env': [{'$class': 'meta_mb.envs.mujoco.walker2d_env.Walker2DEnv'}]}


    for k, v in values.items():
        if params[k] not in v:
            return False
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument('--max_path_length', '-l', type=int, default=None,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', '-n', type=int, default=5,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--gap_pkl', type=int, default=1,
                        help='Gap between pkl policies')
    parser.add_argument('--max_pkl', type=int, default=None,
                        help='Maximum value of the pkl policies')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    parser.add_argument('--ignore_done', action='store_true',
                        help='Whether stop animation when environment done or continue anyway')
    parser.add_argument('--stochastic', action='store_true', help='Apply stochastic action instead of deterministic')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]

    experimet_paths = load_exps_data(args.path, gap=args.gap_pkl, max=args.max_pkl)
    for exp_path in experimet_paths:
        max_path_length = exp_path['json']['max_path_length'] if args.max_path_length is None else args.max_path_length
        if valid_experiment(exp_path['json']):
            for pkl_path in exp_path['pkl']:
                with tf.Session() as sess:
                    print("\n Testing policy %s \n" % pkl_path)
                    data = joblib.load(pkl_path)
                    policy = data['policy']
                    if hasattr(policy, 'switch_to_pre_update'):
                        policy.switch_to_pre_update()
                    env = data['env']
                    video_filename = pkl_path.split('.')[0] + '.mp4'
                    paths = rollout(env, policy, max_path_length=max_path_length, animated=False, speedup=args.speedup,
                                    video_filename=video_filename, save_video=True, ignore_done=args.ignore_done,
                                        stochastic=args.stochastic, num_rollouts=args.num_rollouts)
                    print('Average Returns: ', np.mean([sum(path['rewards']) for path in paths]))
                tf.reset_default_graph()



