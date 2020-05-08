import joblib
import tensorflow as tf
import argparse
import scipy.io as sio
import os.path as osp

import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str)
    parser.add_argument("save_dir", type=str)
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
        weights = policy.get_param_values()
        w0, b0, w1, b1, w2, b2, _ = weights.values()
        sio.savemat(osp.join(args.save_dir, 'policy_weights.mat'),
                    {'w0':w0, 'b0':b0, 'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2})

        x = np.ones([1, 40])

        f1 = np.tanh(np.matmul(x, w0) + b0)
        f2 = np.tanh(np.matmul(f1, w1) + b1)
        f3 = np.matmul(f2, w2) + b2
        actions, agent_infos = policy.get_actions(x)

        assert np.sum(np.abs(f3 - agent_infos[0]['mean'])) < 1e-3


