import joblib
import tensorflow as tf
import argparse
import numpy as np
from meta_mb.envs.blue.real_blue_arm_env import ArmReacherEnv
from meta_mb.samplers.utils import rollout
from meta_mb.envs.normalized_env import normalize


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str)
    parser.add_argument('--max_path_length', '-l', type=int, default=1000,
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
        env = normalize(ArmReacherEnv(side='right'))

        real_rewards = np.array([])
        act_rewards = np.array([])
        pos_rewards = np.array([])
        for _ in range(args.num_rollouts):
            path = rollout(env, policy, max_path_length=args.max_path_length, animated=False, speedup=args.speedup,
                           video_filename=args.video_filename, save_video=False, ignore_done=args.ignore_done,
                           stochastic=args.stochastic)

            real_rewards = np.append(real_rewards, np.sum(path[0]['rewards']))
            print("Real Reward Sum", np.sum(path[0]['rewards']))
            #print(np.mean(path[0]['rewards']))
            #print(len(path_act[0]['rewards']))
            #print(len(path_pos[0]['rewards']))
        print("Real Reward Avg")
        print(np.mean(real_rewards))
