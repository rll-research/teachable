import os
import json
import pickle
import numpy as np
from tensorflow import tanh, ConfigProto
from multiprocessing import Process, Pipe
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.mb_envs import HalfCheetahEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.trainers.parallel_metrpo_trainer import ParallelTrainer
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.logger import logger

INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'parallel_half_cheetah'


def init_vars(sender, config, policy, dynamics_model):
    import tensorflow as tf

    with tf.Session(config=config).as_default() as sess:

        # initialize uninitialized vars  (only initialize vars that were not loaded)
        uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
        sess.run(tf.variables_initializer(uninit_vars))

        policy_pickle = pickle.dumps(policy)
        dynamics_model_pickle = pickle.dumps(dynamics_model)

    sender.send((policy_pickle, dynamics_model_pickle))
    sender.close()


def run_experiment(**kwargs):

    exp_dir = os.getcwd() + '/data/parallel_mb_ppo/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
    print("\n---------- running experiment {} ---------------------------".format(exp_dir))
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)

    # Instantiate classes
    set_seed(kwargs['seed'])

    baseline = kwargs['baseline']()

    env = normalize(kwargs['env']()) # Wrappers?

    policy = GaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        hidden_sizes=kwargs['policy_hidden_sizes'],
        learn_std=kwargs['policy_learn_std'],
        hidden_nonlinearity=kwargs['policy_hidden_nonlinearity'],
        output_nonlinearity=kwargs['policy_output_nonlinearity'],
    )

    dynamics_model = MLPDynamicsEnsemble(
        'dynamics-ensemble',
        env=env,
        num_models=kwargs['num_models'],
        hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
        hidden_sizes=kwargs['dynamics_hidden_sizes'],
        output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
        learning_rate=kwargs['dynamics_learning_rate'],
        batch_size=kwargs['dynamics_batch_size'],
        buffer_size=kwargs['dynamics_buffer_size'],
    )

    '''-------- dumps and reloads -----------------'''

    baseline_pickle = pickle.dumps(baseline)
    env_pickle = pickle.dumps(env)

    receiver, sender = Pipe()
    p = Process(
        target=init_vars,
        name="init_vars",
        args=(sender, config, policy, dynamics_model),
        daemon=True,
    )
    p.start()
    policy_pickle, dynamics_model_pickle = receiver.recv()
    receiver.close()

    '''-------- following classes depend on baseline, env, policy, dynamics_model -----------'''
    
    worker_data_feed_dict = {
        'env_sampler': {
            'num_rollouts': kwargs['num_rollouts'],
            'max_path_length': kwargs['max_path_length'],
            'n_parallel': kwargs['n_parallel'],
        },
        'dynamics_sample_processor': {
            'discount': kwargs['discount'],
            'gae_lambda': kwargs['gae_lambda'],
            'normalize_adv': kwargs['normalize_adv'],
            'positive_adv': kwargs['positive_adv'],
        },
    }

    worker_model_feed_dict = {}
    
    worker_policy_feed_dict = {
        'model_sampler': {
            'num_rollouts': kwargs['imagined_num_rollouts'],
            'max_path_length': kwargs['max_path_length'],
            'dynamics_model': dynamics_model,
            'deterministic': kwargs['deterministic'],
        },
        'model_sample_processor': {
            'discount': kwargs['discount'],
            'gae_lambda': kwargs['gae_lambda'],
            'normalize_adv': kwargs['normalize_adv'],
            'positive_adv': kwargs['positive_adv'],
        },
        'algo': {
            'learning_rate': kwargs['learning_rate'],
            'clip_eps': kwargs['clip_eps'],
            'max_epochs': kwargs['num_ppo_steps'],
        }
    }

    trainer = ParallelTrainer(
        policy_pickle=policy_pickle,
        env_pickle=env_pickle,
        baseline_pickle=baseline_pickle,
        dynamics_model_pickle=dynamics_model_pickle,
        feed_dicts=[worker_data_feed_dict, worker_model_feed_dict, worker_policy_feed_dict],
        n_itr=kwargs['n_itr'],
        dynamics_model_max_epochs=kwargs['dynamics_max_epochs'],
        log_real_performance=kwargs['log_real_performance'],
        steps_per_iter=kwargs['steps_per_iter'],
        flags_need_query=kwargs['flags_need_query'],
        config=config,
        simulation_sleep=kwargs['simulation_sleep'],
    )

    trainer.train()


if __name__ == '__main__':

    sweep_params = {

        'flags_need_query': [
            [False, False, False],
            #[True, True, False], [False, True, True], [True, False, True],
            #[True, False, False], [False, True, False], [False, False, True],
            [True, True, True],
        ],

        'seed': [1, 2, 3],

        'algo': ['meppo'],
        'baseline': [LinearFeatureBaseline],
        'env': [HalfCheetahEnv],

        # Problem Conf
        'n_itr': [1000],
        'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],
        'steps_per_iter': [(5, 5)],

        # Real Env Sampling
        'num_rollouts': [5, 10],
        'n_parallel': [1],
        'simulation_sleep': [10, 20],

        # Dynamics Model
        'num_models': [5],
        'dynamics_hidden_sizes': [(512, 512)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [5, 35],
        'dynamics_learning_rate': [8e-3, 2e-4, 2e-3],
        'dynamics_batch_size': [128,],
        'dynamics_buffer_size': [10000],
        'deterministic': [False],

        # Policy
        'policy_hidden_sizes': [(64, 64)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tanh],
        'policy_output_nonlinearity': [None],

        # Algo
        'clip_eps': [0.3],
        'learning_rate': [1e-3],
        'num_ppo_steps': [5],
        'imagined_num_rollouts': [20],
        'scope': [None],
        'exp_tag': ['parallel_half_cheetah'],  # For changes besides hyperparams

    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

