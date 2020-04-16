from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble
from meta_mb.trainers.parallel_mb_trainer import ParallelTrainer
from meta_mb.policies.mpc_controller import MPCController
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import ClassEncoder, set_seed
from meta_mb.envs.mb_envs import AntEnv, Walker2dEnv, HalfCheetahEnv, HopperEnv
import json
import os
from multiprocessing import Process, Pipe
from tensorflow import ConfigProto
import pickle

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = '2x-halfcheetah-ant-det-mbmpc'

def init_vars(sender, config_sess, policy, dynamics_model):
    import tensorflow as tf

    with tf.Session(config=config_sess).as_default() as sess:

        # initialize uninitialized vars  (only initialize vars that were not loaded)
        uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
        sess.run(tf.variables_initializer(uninit_vars))

        policy_pickle = pickle.dumps(policy)
        dynamics_model_pickle = pickle.dumps(dynamics_model)

    sender.send((policy_pickle, dynamics_model_pickle))
    sender.close()


def run_experiment(**config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '-' + config.get('exp_name', '')
    print("\n---------- experiment with dir {} ---------------------------".format(exp_dir))
    logger.configure(dir=exp_dir, format_strs=['csv', 'stdout', 'log'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    os.makedirs(exp_dir + '/Data/', exist_ok=True)
    os.makedirs(exp_dir + '/Model/', exist_ok=True)
    os.makedirs(exp_dir + '/Policy/', exist_ok=True)
    json.dump(config, open(exp_dir + '/Data/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(config, open(exp_dir + '/Model/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(config, open(exp_dir + '/Policy/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    run_base(exp_dir, **config)


def run_base(exp_dir, **config):

    config_sess = ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)

    # Instantiate classes
    set_seed(config['seed'])

    if config['env'] == 'Ant':
        env = AntEnv()
        simulation_sleep = 0.05 * config['num_rollouts'] * config['max_path_length'] * config['simulation_sleep_frac']
    elif config['env'] == 'HalfCheetah':
        env = HalfCheetahEnv()
        simulation_sleep = 0.05 * config['num_rollouts'] * config['max_path_length'] * config['simulation_sleep_frac']
    elif config['env'] == 'Hopper':
        env = HopperEnv()
        simulation_sleep = 0.008 * config['num_rollouts'] * config['max_path_length'] * config['simulation_sleep_frac']
    elif config['env'] == 'Walker2d':
        env = Walker2dEnv()
        simulation_sleep = 0.008 * config['num_rollouts'] * config['max_path_length'] * config['simulation_sleep_frac']
    else:
        raise NotImplementedError

    if config['probabilistic_dynamics']:
        dynamics_model = ProbMLPDynamicsEnsemble(
            'prob-dynamics-ensemble',
            env=env,
            num_models=config['num_models'],
            hidden_nonlinearity=config['dynamics_hidden_nonlinearity'],
            hidden_sizes=config['dynamics_hidden_sizes'],
            output_nonlinearity=config['dynamics_output_nonlinearity'],
            learning_rate=config['dynamics_learning_rate'],
            batch_size=config['dynamics_batch_size'],
            buffer_size=config['dynamics_buffer_size'],
            rolling_average_persitency=config['rolling_average_persitency']
        )
    else:
        dynamics_model = MLPDynamicsEnsemble(
            'dynamics-ensemble',
            env=env,
            num_models=config['num_models'],
            hidden_nonlinearity=config['dynamics_hidden_nonlinearity'],
            hidden_sizes=config['dynamics_hidden_sizes'],
            output_nonlinearity=config['dynamics_output_nonlinearity'],
            learning_rate=config['dynamics_learning_rate'],
            batch_size=config['dynamics_batch_size'],
            buffer_size=config['dynamics_buffer_size'],
            rolling_average_persitency=config['rolling_average_persitency']
        )

    policy = MPCController(
        name="policy",
        env=env,
        dynamics_model=dynamics_model,
        discount=config['discount'],
        n_candidates=config['n_candidates'],
        horizon=config['horizon'],
        use_cem=config['use_cem'],
        num_cem_iters=config['num_cem_iters'],
    )

    '''-------- dumps and reloads -----------------'''

    env_pickle = pickle.dumps(env)

    receiver, sender = Pipe()
    p = Process(
        target=init_vars,
        name="init_vars",
        args=(sender, config_sess, policy, dynamics_model),
        daemon=False,
    )
    p.start()
    policy_pickle, dynamics_model_pickle = receiver.recv()
    receiver.close()

    '''-------- following classes depend on baseline, env, policy, dynamics_model -----------'''

    worker_data_feed_dict = {
        'sampler': {
            'num_rollouts': config['num_rollouts'],
            'max_path_length': config['max_path_length'],
            'n_parallel': config['n_parallel'],
        },
        'sample_processor': {},
    }

    worker_model_feed_dict = {}

    trainer = ParallelTrainer(
        exp_dir=exp_dir,
        env_pickle=env_pickle,
        policy_pickle=policy_pickle,
        baseline_pickle=None,
        dynamics_model_pickle=dynamics_model_pickle,
        feed_dicts=[worker_data_feed_dict, worker_model_feed_dict],
        n_itr=config['n_itr'],
        initial_random_samples=config['initial_random_samples'],
        initial_sinusoid_samples=config['initial_sinusoid_samples'],
        flags_need_query=config['flags_need_query'],
        config=config_sess,
        simulation_sleep=simulation_sleep,
    )

    trainer.train()


if __name__ == '__main__':

    config = {

        'flags_need_query': [
            [False, False, False],
            # [True, True, True],
        ],
        'rolling_average_persitency': [0.1, 0.4],

        'seed': [1, 2],

        'n_itr': [101*20],
        'num_rollouts': [1],
        'simulation_sleep_frac': [1],
        'env': ['HalfCheetah', 'Ant'], #['Walker2d', 'Hopper', 'Ant', 'HalfCheetah', ],

        # Problem
        'probabilistic_dynamics': [False],
        'max_path_length': [50],
        'normalize': [False],
        'discount': [1.],

        # Policy
        'n_candidates': [1000], # K ###
        'horizon': [20], # Tau ###
        'use_cem': [False],
        'num_cem_iters': [5],

        # Training
        'dynamics_learning_rate': [5e-4, 0.001],
        'valid_split_ratio': [0.1],
        'initial_random_samples': [True],
        'initial_sinusoid_samples': [False],

        # Dynamics Model
        'recurrent': [False],
        'num_models': [5],
        'dynamics_hidden_nonlinearity': ['swish'],
        'dynamics_output_nonlinearity': [None],
        'dynamics_hidden_sizes': [(512, 512, 512)],
        'dynamic_model_epochs': [50],  # UNUSED
        'dynamics_buffer_size': [25000],
        'backprop_steps': [100],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'dynamics_batch_size': [64],
        'cell_type': ['lstm'],

        #  Other
        'n_parallel': [1],
        'exp_tag': ['parallel-mbmpc']
    }

    assert config['n_candidates'][0] % config['num_models'][0] == 0  # FIXME: remove constraint

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
