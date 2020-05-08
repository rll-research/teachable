import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.mujoco.walker2d_env import Walker2DEnv
from meta_mb.envs.mb_envs import AntEnv, Walker2dEnv, HalfCheetahEnv
from meta_mb.envs.mujoco.hopper_env import HopperEnv
# from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
from meta_mb.envs.pr2.new_real_pr2_env import PR2Env
from meta_mb.envs.pr2.real_pr2_reach_env import PR2ReachEnv
from meta_mb.envs.pr2.real_pr2_peg_env import PR2PegEnv
from meta_mb.envs.pr2.real_pr2_stack_env import PR2StackEnv
from meta_mb.envs.pr2.real_pr2_water_bottle import PR2BottleEnv
from meta_mb.meta_algos.trpo_maml import TRPOMAML
from meta_mb.trainers.mbmpo_trainer import Trainer
from meta_mb.samplers.meta_samplers.single_meta_sampler import SingleMetaSampler
from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.samplers.mbmpo_samplers.mbmpo_sampler import MBMPOSampler
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.logger import logger
from meta_mb.envs.normalized_env import normalize

INSTANCE_TYPE = 'c4.4xlarge'
EXP_NAME = 'mbmpo_sequential_peg_video_0'

class Num:
    EXP_NUM = 0


def run_experiment(**kwargs):

    num = Num()
    exp_name = EXP_NAME + str(num.EXP_NUM)

    exp_dir = os.getcwd() + '/data/video_peg/' + EXP_NAME + kwargs.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['csv', 'stdout', 'log'], snapshot_mode='all') #change to all
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    Num.EXP_NUM += 1
    with sess.as_default() as sess:

        # Instantiate classesLogger
        set_seed(kwargs['seed'])

        baseline = kwargs['baseline']()

        env = normalize(kwargs['env']()) # Wrappers?

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

        env_sampler = BaseSampler(
            env=env,
            policy=policy,
            # rollouts_per_meta_task=kwargs['real_env_rollouts_per_meta_task'],
            num_rollouts=kwargs['meta_batch_size'],
            max_path_length=kwargs['max_path_length'],
            sleep_reset=2.5,
            #parallel=kwargs['parallel'],
            # parallel=False
        )


        model_sampler = MBMPOSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=kwargs['rollouts_per_meta_task'],
            meta_batch_size=kwargs['meta_batch_size'],
            max_path_length=kwargs['max_path_length'],
            dynamics_model=dynamics_model,
            deterministic=kwargs['deterministic'],
        )

        dynamics_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        model_sample_processor = MAMLSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        algo = TRPOMAML(
            policy=policy,
            step_size=kwargs['step_size'],
            inner_type=kwargs['inner_type'],
            inner_lr=kwargs['inner_lr'],
            meta_batch_size=kwargs['meta_batch_size'],
            num_inner_grad_steps=kwargs['num_inner_grad_steps'],
            exploration=kwargs['exploration'],
        )

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=env,
            model_sampler=model_sampler,
            env_sampler=env_sampler,
            model_sample_processor=model_sample_processor,
            dynamics_sample_processor=dynamics_sample_processor,
            dynamics_model=dynamics_model,
            n_itr=kwargs['n_itr'],
            num_inner_grad_steps=kwargs['num_inner_grad_steps'],
            dynamics_model_max_epochs=kwargs['dynamics_max_epochs'],
            log_real_performance=kwargs['log_real_performance'],
            meta_steps_per_iter=kwargs['meta_steps_per_iter'],
            sample_from_buffer=True,
            sess=sess,
        )

        trainer.train()




if __name__ == '__main__':

    sweep_params = {
        'seed': [2],
        'algo': ['mbmpo'],
        'baseline': [LinearFeatureBaseline],
        'env': [PR2ReachEnv],

        # Problem Conf
        'n_itr': [20], #previously 101


        'max_path_length': [15], #previously 200
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [False],
        'meta_steps_per_iter': [(30, 30)],

        # Real Env Sampling)
        'real_env_rollouts_per_meta_task': [1],
        'parallel': [True],
        'fraction_meta_batch_size': [1.],

        # Dynamics Model
        'num_models': [5],
        'dynamics_hidden_sizes': [(512, 512)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [200],
        'dynamics_learning_rate': [5e-4],
        'dynamics_batch_size': [256],
        'dynamics_buffer_size': [10000],
        'deterministic': [True],


        # Policy
        'policy_hidden_sizes': [(32, 32)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tf.tanh],
        'policy_output_nonlinearity': [None],

        # Meta-Algo
        'meta_batch_size': [5],  # Note: It has to be multiple of num_models #previously 20
        'rollouts_per_meta_task': [50], #previously 50
        'num_inner_grad_steps': [1],
        'inner_lr': [0.001],
        'inner_type': ['log_likelihood'],
        'step_size': [0.01],
        'exploration': [False],
        'sample_from_buffer': [True],

        'scope': [None],
        'exp_tag': ['mbmpo_all'], # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

