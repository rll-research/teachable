import os
import json
import pickle
import numpy as np
from tensorflow import tanh, ConfigProto
from multiprocessing import Process, Pipe
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.mb_envs import AntEnv, Walker2dEnv, HalfCheetahEnv, HopperEnv
from meta_mb.trainers.parallel_mbmpo_trainer import ParallelTrainer
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.logger import logger

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'bptt-a-mb-mpo-5'


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
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
    print("\n---------- experiment with dir {} ---------------------------".format(exp_dir))
    logger.configure(dir=exp_dir, format_strs=['csv', 'stdout', 'log'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    os.makedirs(exp_dir + '/Data/', exist_ok=True)
    os.makedirs(exp_dir + '/Model/', exist_ok=True)
    os.makedirs(exp_dir + '/Policy/', exist_ok=True)
    json.dump(kwargs, open(exp_dir + '/Data/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(kwargs, open(exp_dir + '/Model/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(kwargs, open(exp_dir + '/Policy/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    run_base(exp_dir, **kwargs)


def run_base(exp_dir, **kwargs):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)

    # Instantiate classes
    set_seed(kwargs['seed'])

    baseline = kwargs['baseline']()

    if kwargs['env'] == 'Ant':
        env = AntEnv()
        simulation_sleep = 0.05 * kwargs['num_rollouts'] * kwargs['max_path_length'] * kwargs['simulation_sleep_frac']
    elif kwargs['env'] == 'HalfCheetah':
        env = HalfCheetahEnv()
        simulation_sleep = 0.05 * kwargs['num_rollouts'] * kwargs['max_path_length'] * kwargs['simulation_sleep_frac']
    elif kwargs['env'] == 'Hopper':
        env = HopperEnv()
        simulation_sleep = 0.008 * kwargs['num_rollouts'] * kwargs['max_path_length'] * kwargs['simulation_sleep_frac']
    elif kwargs['env'] == 'Walker2d':
        env = Walker2dEnv()
        simulation_sleep = 0.008 * kwargs['num_rollouts'] * kwargs['max_path_length'] * kwargs['simulation_sleep_frac']
    else:
        raise NotImplementedError

    policy = MetaGaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=kwargs['meta_batch_size'],
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
        rolling_average_persitency=kwargs['rolling_average_persitency'],
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
            'rollouts_per_meta_task': kwargs['real_env_rollouts_per_meta_task'],
            'meta_batch_size': kwargs['meta_batch_size'],
            'max_path_length': kwargs['max_path_length'],
            'parallel': kwargs['parallel'],
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
            'rollouts_per_meta_task': kwargs['rollouts_per_meta_task'],
            'meta_batch_size': kwargs['meta_batch_size'],
            'max_path_length': kwargs['max_path_length'],
            'deterministic': kwargs['deterministic'],
        },
        'model_sample_processor': {
            'discount': kwargs['discount'],
            'gae_lambda': kwargs['gae_lambda'],
            'normalize_adv': kwargs['normalize_adv'],
            'positive_adv': kwargs['positive_adv'],
        },
        'algo': {
            'step_size': kwargs['step_size'],
            'inner_type': kwargs['inner_type'],
            'inner_lr': kwargs['inner_lr'],
            'meta_batch_size': kwargs['meta_batch_size'],
            'num_inner_grad_steps': kwargs['num_inner_grad_steps'],
            'exploration': kwargs['exploration'],
        }
    }

    trainer = ParallelTrainer(
        exp_dir=exp_dir,
        policy_pickle=policy_pickle,
        env_pickle=env_pickle,
        baseline_pickle=baseline_pickle,
        dynamics_model_pickle=dynamics_model_pickle,
        feed_dicts=[worker_data_feed_dict, worker_model_feed_dict, worker_policy_feed_dict],
        n_itr=kwargs['n_itr'],
        num_inner_grad_steps=kwargs['num_inner_grad_steps'],
        initial_random_samples=kwargs['initial_random_samples'],
        flags_need_query=kwargs['flags_need_query'],
        num_rollouts_per_iter=int(kwargs['meta_batch_size'] * kwargs['fraction_meta_batch_size']),
        config=config,
        simulation_sleep=simulation_sleep,
        sampler_str=kwargs['sampler'],
    )

    trainer.train()


if __name__ == '__main__':

    sweep_params = {

        'flags_need_query': [
            [False, False, False],
        ],
        'rolling_average_persitency': [
            0.1, 0.4, 0.9, 0.99
        ],

        'seed': [1, 2, 3, 4],

        'n_itr': [1250],
        'num_rollouts': [1],
        'simulation_sleep_frac': [1],

        'env': ['Hopper', 'Ant'],

        # Problem Conf
        'num_models': [5],
        'algo': ['mbmpo'],
        'sampler': ['bptt'],
        'baseline': [LinearFeatureBaseline],
        'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],  # not implemented
        'meta_steps_per_iter': [1],  # UNUSED

        # Real Env Sampling
        'real_env_rollouts_per_meta_task': [1],
        'parallel': [False],
        'fraction_meta_batch_size': [0.05],

        # Dynamics Model
        'dynamics_hidden_sizes': [(512, 512, 512)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [50],  # UNUSED
        'dynamics_learning_rate': [1e-3, 5e-4],
        'dynamics_batch_size': [256],
        'dynamics_buffer_size': [10000],
        'deterministic': [False],
        'loss_str': ['MSE'],
        'initial_random_samples': [True],

        # Policy
        'policy_hidden_sizes': [(64, 64)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tanh],
        'policy_output_nonlinearity': [None],

        # Meta-Algo
        'meta_batch_size': [20],  # Note: It has to be multiple of num_models
        'rollouts_per_meta_task': [25],
        'num_inner_grad_steps': [1],
        'inner_lr': [0.001],
        'inner_type': ['log_likelihood'],
        'step_size': [0.01, 0.02, 0.05],
        'exploration': [False],
        'sample_from_buffer': [False],  # not implemented

        'scope': [None],
        'exp_tag': ['a-mb-mpo'],  # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

