from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble
from meta_mb.trainers.mb_trainer import Trainer
from meta_mb.policies.mpc_controller import MPCController
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import ClassEncoder
from meta_mb.envs.mb_envs import *
import json
import os
import tensorflow as tf

EXP_NAME = 'mb-mpc-blue-train'

INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    with sess.as_default() as sess:

        env = config['env']()

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

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=config['max_path_length'],
            n_parallel=config['n_parallel'],
        )

        sample_processor = ModelSampleProcessor()

        algo = Trainer(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            sampler=sampler,
            dynamics_sample_processor=sample_processor,
            n_itr=config['n_itr'],
            initial_random_samples=config['initial_random_samples'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            sess=sess,
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
                'seed': [5],

                # Problem
                'env': [HalfCheetahEnv],  # 'HalfCheetahEnv'
                'max_path_length': [200],
                'normalize': [False],
                 'n_itr': [50],
                'discount': [1.],

                # Policy
                'n_candidates': [2000], # K
                'horizon': [20], # Tau
                'use_cem': [False],
                'num_cem_iters': [5],

                # Training
                'num_rollouts': [5],
                'dynamics_learning_rate': [0.001],
                'valid_split_ratio': [0.1],
                'rolling_average_persitency': [0.4],
                'initial_random_samples': [True],
                'initial_sinusoid_samples': [False],

                # Dynamics Model
                'probabilistic_dynamics': [True],
                'recurrent': [False],
                'num_models': [5],
                'dynamics_hidden_nonlinearity': ['swish'],
                'dynamics_output_nonlinearity': [None],
                'dynamics_hidden_sizes': [(512, 512, 512)],
                'dynamic_model_epochs': [50],
                'dynamics_buffer_size': [25000],
                'backprop_steps': [100],
                'weight_normalization_model': [False],  # FIXME: Doesn't work
                'dynamics_batch_size': [64],
                'cell_type': ['lstm'],

                #  Other
                'n_parallel': [5],

    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
