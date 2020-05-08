import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.normalized_env import normalize
from meta_mb.envs.img_wrapper_env import image_wrapper
from meta_mb.optimizers.random_search_optimizer import RandomSearchOptimizer
from meta_mb.trainers.ars_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.ars_sampler.ars_sample_processor import ARSSamplerProcessor
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.samplers.ars_sampler.ars_sampler import ARSSampler
from meta_mb.policies.np_nn_policy import NNPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.logger import logger
from meta_mb.unsupervised_learning.vae import VAE
from meta_mb.envs.mb_envs import *
INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'ars-mod-buff-2'


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
            env = normalize(kwargs['env']())

        else:
            vae = VAE(latent_dim=8)
            env = image_wrapper(normalize(kwargs['env']()), vae=vae, latent_dim=32)

        policy = NNPolicy(name="policy",
                          obs_dim=np.prod(env.observation_space.shape),
                          action_dim=np.prod(env.action_space.shape),
                          hidden_sizes=kwargs['hidden_sizes'],
                          normalization=None,
                          )

        dynamics_model = MLPDynamicsEnsemble('dynamics-ensemble',
                                             env=env,
                                             num_models=kwargs['num_models'],
                                             hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                             hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                             output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                             learning_rate=kwargs['dynamics_learning_rate'],
                                             batch_size=kwargs['dynamics_batch_size'],
                                             buffer_size=kwargs['dynamics_buffer_size'],
                                             )

        # dynamics_model = None
        assert kwargs['rollouts_per_policy'] % kwargs['num_models'] == 0

        env_sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            n_parallel=kwargs['num_rollouts'],
        )

        # TODO: I'm not sure if it works with more than one rollout per model

        model_sampler = ARSSampler(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            rollouts_per_policy=kwargs['rollouts_per_policy'],
            max_path_length=kwargs['horizon'],
            num_deltas=kwargs['num_deltas'],
            n_parallel=1,
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
            uncertainty_coeff=kwargs['uncertainty_coeff']
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
            dynamics_model=dynamics_model,
            num_deltas=kwargs['num_deltas'],
            n_itr=kwargs['n_itr'],
            dynamics_model_max_epochs=kwargs['dynamics_max_epochs'],
            log_real_performance=kwargs['log_real_performance'],
            steps_per_iter=kwargs['steps_per_iter'],
            delta_std=kwargs['delta_std'],
            sess=sess,
            initial_random_samples=True,
            sample_from_buffer=kwargs['sample_from_buffer']
        )

        trainer.train()


if __name__ == '__main__':

    sweep_params = {
        'seed': [1, 2],

        'algo': ['ars'],
        'baseline': [LinearFeatureBaseline],
        'env': [HalfCheetahEnv],
        'use_images': [False],

        # Problem Conf
        'n_itr': [200],
        'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],
        'steps_per_iter': [(20, 20)],

        # Real Env Sampling
        'num_rollouts': [5],
        'parallel': [True],

        # Dynamics Model
        'num_models': [5],
        'dynamics_hidden_sizes': [(500, 500, 500)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [50],
        'dynamics_learning_rate': [1e-3],
        'dynamics_batch_size': [128],
        'dynamics_buffer_size': [25000],

        # Meta-Algo
        'learning_rate': [0.001],
        'horizon': [200],
        'num_deltas': [4],
        'rollouts_per_policy': [5],
        'percentile': [.0],
        'delta_std': [0.03],
        'hidden_sizes': [(64, 64)],
        'sample_from_buffer': [False],
        'uncertainty_coeff': [.1],

        'scope': [None],
        'exp_tag': [''],  # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

