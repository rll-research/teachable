import joblib
import tensorflow as tf
import argparse
import numpy as np
from meta_mb.samplers.utils import rollout
from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
from meta_mb.envs.blue.real_blue_arm_env import ArmReacherEnv
from meta_mb.envs.blue.full_blue_env import FullBlueEnv
from meta_mb.envs.blue.blue_env import BlueEnv
from meta_mb.envs.blue.mimic_blue_pos_env import MimicBluePosEnv
from meta_mb.envs.blue.mimic_blue_action_env import MimicBlueActEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.meta_envs.rl2_env import rl2env
import time
import pickle



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str)
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', '-n', type=int, default=10,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
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
    with tf.Session() as sess:
        pkl_path = args.param
        print("Testing policy %s" % pkl_path)
        data = joblib.load(pkl_path)
        policy = data['policy']
        env = data['env']
        real_rewards = np.array([])
        act_rewards = np.array([])
        pos_rewards = np.array([])
        for i in range(args.num_rollouts):
            path = rollout(env, policy, max_path_length=args.max_path_length, animated=True, speedup=args.speedup,
                           video_filename=args.video_filename, save_video=False, ignore_done=args.ignore_done,
                           stochastic=args.stochastic)

            print(env.actions)
            pickle.dump(env.actions, open("actions_0.pkl", "wb"))
