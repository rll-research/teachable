import os
import json
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.envs.mb_envs import *
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.svg_1 import SVG1
from meta_mb.trainers.svg_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics import MLPDynamicsModel
from meta_mb.baselines.nn_basline import NNValueFun
from meta_mb.logger import logger
import tensorflow as tf

INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'svg-test-nn'


def run_experiment(**kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)

    with sess.as_default() as sess:
        exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
        json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

        # Instantiate classes
        set_seed(kwargs['seed'])

        env = normalize(kwargs['env']()) # Wrappers?

        baseline = NNValueFun('value-function',
                              env,
                              hidden_nonlinearity=kwargs['vfun_hidden_nonlinearity'],
                              hidden_sizes=kwargs['vfun_hidden_sizes'],
                              output_nonlinearity=kwargs['vfun_output_nonlinearity'],
                              learning_rate=kwargs['vfun_learning_rate'],
                              batch_size=kwargs['vfun_batch_size'],
                              buffer_size=kwargs['vfun_buffer_size'],
                              normalize_input=False,
                              )

        policy = GaussianMLPPolicy(
            name="policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=kwargs['policy_hidden_sizes'],
            learn_std=kwargs['policy_learn_std'],
            output_nonlinearity=kwargs['policy_output_nonlinearity'],
        )

        dynamics_model = MLPDynamicsModel('prob-dynamics',
                                         env=env,
                                         hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                         hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                         output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                         learning_rate=kwargs['dynamics_learning_rate'],
                                         batch_size=kwargs['dynamics_batch_size'],
                                         buffer_size=kwargs['dynamics_buffer_size'],
                                         normalize_input=False,
                                         )

        assert kwargs['num_rollouts'] % kwargs['n_parallel'] == 0

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            n_parallel=kwargs['n_parallel'],
        )

        sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        algo = SVG1(
            policy=policy,
            dynamics_model=dynamics_model,
            value_function=baseline,
            tf_reward=env.tf_reward,
            learning_rate=kwargs['svg_learning_rate'],
            num_grad_steps=kwargs['num_rollouts'] * kwargs['max_path_length']//kwargs['svg_batch_size'],
            batch_size=kwargs['svg_batch_size'],
            discount=kwargs['discount'],
            kl_penalty=kwargs['kl_penalty'],
        )

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=env,
            sampler=sampler,
            sample_processor=sample_processor,
            dynamics_model=dynamics_model,
            value_function=baseline,
            n_itr=kwargs['n_itr'],
            dynamics_model_max_epochs=kwargs['dynamics_max_epochs'],
            vfun_max_epochs=kwargs['vfun_max_epochs'],
            sess=sess,
        )

        trainer.train()


if __name__ == '__main__':

    sweep_params = {
        'seed': [1, 2, 3, 4],

        'algo': ['svg'],
        'env': [PendulumO01Env, PendulumO001Env],

        # Problem Conf
        'n_itr': [100],
        'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],

        # Env Sampling
        'num_rollouts': [10],
        'n_parallel': [5],

        # Dynamics Model
        'dynamics_hidden_sizes': [(500, 500)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [50],
        'dynamics_learning_rate': [1e-3],
        'dynamics_batch_size': [256],
        'dynamics_buffer_size': [200 * 5 * 10],

        # Value Function
        'vfun_hidden_sizes': [(200, 200)],
        'vfun_hidden_nonlinearity': ['swish'],
        'vfun_output_nonlinearity': [None],
        'vfun_max_epochs': [50],
        'vfun_learning_rate': [1e-4],
        'vfun_batch_size': [32],
        'vfun_buffer_size': [10000],


        # Policy
        'policy_hidden_sizes': [(100, 100)],
        'policy_learn_std': [True],
        'policy_output_nonlinearity': [None],

        # Algo
        'svg_learning_rate': [0.0001],
        'svg_batch_size': [64],
        'svg_max_buffer_size': [200 * 5 * 5],
        'kl_penalty': [.1],

        'scope': [None],
        'exp_tag': [''], # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

