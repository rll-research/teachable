import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.cassie.cassie_env import CassieEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.envs.img_wrapper_env import image_wrapper
# from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
from meta_mb.optimizers.random_search_optimizer import RandomSearchOptimizer
from meta_mb.trainers.ars_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.ars_sampler.ars_sample_processor import ARSSamplerProcessor
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.samplers.ars_sampler.ars_sampler import ARSSampler
from meta_mb.policies.np_nn_policy import NNPolicy
from meta_mb.logger import logger
from meta_mb.unsupervised_learning.vae import VAE

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'cassie-running-2'


def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    with sess.as_default() as sess:

        # Instantiate classes
        set_seed(kwargs['seed'])

        baseline = kwargs['baseline']()

        if not kwargs['use_images']:
            env = normalize(kwargs['env'](policytask=kwargs['task']))
            vae = None

        else:
            vae = VAE(latent_dim=kwargs['latent_dim'], channels=3 * kwargs['time_steps'])
            env = image_wrapper(normalize(kwargs['env']()),
                                latent_dim=kwargs['latent_dim'],
                                time_steps=kwargs['time_steps'])

        policy = NNPolicy(name="policy",
                          obs_dim=np.prod(env.observation_space.shape),
                          action_dim=np.prod(env.action_space.shape),
                          hidden_sizes=kwargs['hidden_sizes'],
                          normalization=kwargs['normalization'],
                          )

        env_sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            vae=vae,
        )

        model_sampler = ARSSampler(
            env=env,
            policy=policy,
            rollouts_per_policy=kwargs['rollouts_per_policy'],
            max_path_length=kwargs['max_path_length'],
            num_deltas=kwargs['num_deltas'],
            n_parallel=kwargs['num_deltas'],
            vae=vae,
        )

        dynamics_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        ars_sample_processor = ARSSamplerProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        algo = RandomSearchOptimizer(
            policy=policy,
            learning_rate=kwargs['learning_rate'],
            num_deltas=kwargs['num_deltas'],
            percentile=kwargs['percentile']
        )

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=env,
            model_sampler=model_sampler,
            env_sampler=env_sampler,
            ars_sample_processor=ars_sample_processor,
            dynamics_sample_processor=dynamics_sample_processor,
            num_deltas=kwargs['num_deltas'],
            n_itr=kwargs['n_itr'],
            log_real_performance=kwargs['log_real_performance'],
            steps_per_iter=kwargs['steps_per_iter'],
            delta_std=kwargs['delta_std'],
            sess=sess
        )

        trainer.train()


if __name__ == '__main__':

    sweep_params = {
        'seed': [1, 2],

        'algo': ['ars'],
        'baseline': [LinearFeatureBaseline],
        'env': [CassieEnv],
        'use_images': [False],

        # Problem Conf
        'n_itr': [40],
        'max_path_length': [500],
        'discount': [1.],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],
        'steps_per_iter': [(10, 10)],

        # Real Env Sampling
        'num_rollouts': [2],
        'parallel': [True],

        # Meta-Algo
        'learning_rate': [0.02, 0.01, 0.005],
        'num_deltas': [16],
        'rollouts_per_policy': [1],
        'percentile': [0.],
        'delta_std': [0.005, 0.002],
        'latent_dim': [32],
        'hidden_sizes': [(256, 256, 256)],
        'time_steps': [1],
        'normalization': [None],
        'task': ['balancing'],

        'scope': [None],
        'exp_tag': [''], # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

