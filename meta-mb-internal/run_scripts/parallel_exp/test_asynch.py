import os
import json
import pickle
import numpy as np
from tensorflow import tanh, ConfigProto
from multiprocessing import Process, Pipe
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.logger import logger
from run_scripts.parallel_exp.parallel_mbppo_run_sweep import run_base


INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'test-20-rollouts'


def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
    print("\n---------- experiment with dir {} ---------------------------".format(exp_dir))
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    os.makedirs(exp_dir + '/Data/', exist_ok=True)
    os.makedirs(exp_dir + '/Model/', exist_ok=True)
    os.makedirs(exp_dir + '/Policy/', exist_ok=True)
    json.dump(kwargs, open(exp_dir + '/Data/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(kwargs, open(exp_dir + '/Model/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(kwargs, open(exp_dir + '/Policy/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)

    run_base(exp_dir, **kwargs)


if __name__ == '__main__':

    sweep_params = {

        'flags_need_query': [
            [False, False, False],
            # [True, True, True],
        ],
        'flags_push_freq': [
            #[20, 1, 20],
            #[20, 1, 1],
            #[1, 1, 20],
            [1, 1, 1],
        ],
        'flags_pull_freq': [
            [1, 1, 1],
        ],

        'rolling_average_persitency': [0.99],

        'seed': [1,],
        'probabilistic_dynamics': [False],
        'num_models': [5],

        'n_itr': [501], # num_samples = num_rollouts * max_path_length * n_itr
        'num_rollouts': [20],
        'simulation_sleep_frac': [2, 1, 0.5],
        'env': ['HalfCheetah', 'Ant', 'Walker2d', 'Hopper'],

        # Problem Conf

        'algo': ['meppo'],
        'baseline': [LinearFeatureBaseline],
        'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],  # UNUSED
        'steps_per_iter': [1],  # UNUSED

        # Real Env Sampling
        'n_parallel': [1],

        # Dynamics Model
        'dynamics_hidden_sizes': [(512, 512, 512)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [50],  # UNUSED
        'dynamics_learning_rate': [1e-3],
        'dynamics_batch_size': [256,],
        'dynamics_buffer_size': [25000],
        'deterministic': [False],
        'initial_random_samples': [True],
        'loss_str': ['MSE'],

        # Policy
        'policy_hidden_sizes': [(64, 64)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tanh],
        'policy_output_nonlinearity': [None],

        # Algo
        'clip_eps': [0.3],
        'learning_rate': [1e-3],
        'num_ppo_steps': [5],
        'imagined_num_rollouts': [50,],
        'scope': [None],
        'exp_tag': ['parallel-mbppo'],  # For changes besides hyperparams

    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

