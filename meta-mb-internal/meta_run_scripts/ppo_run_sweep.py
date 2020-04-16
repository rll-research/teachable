import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.meta_envs.mujoco.ant_rand_direc import AntRandDirecEnv
from meta_mb.meta_envs.mujoco.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_mb.meta_envs.mujoco.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_mb.meta_envs.mujoco.humanoid_rand_direc import HumanoidRandDirecEnv
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv
from meta_mb.meta_envs.mujoco.ant_rand_goal import AntRandGoalEnv
from rand_param_envs.pr2_env_reach import PR2Env
from meta_mb.meta_envs.mujoco.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
from meta_mb.envs.jelly.walk_jelly import WalkJellyEnv
from meta_mb.envs.blue.blue_env import BlueEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.meta_algos.ppo_maml import PPOMAML
from meta_mb.trainers.meta_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_mb.logger import logger

INSTANCE_TYPE = 'c4.4xlarge'
EXP_NAME = 'promp-reach-blue-right'


def run_experiment(**config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last_gap', snapshot_gap=50)
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    # Instantiate classes
    set_seed(config['seed'])

    baseline = config['baseline']()

    env = normalize(config['env']()) # Wrappers?

    policy = MetaGaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=config['meta_batch_size'],
        hidden_sizes=config['hidden_sizes'],
        learn_std=config['learn_std'],
        hidden_nonlinearity=config['hidden_nonlinearity'],
        output_nonlinearity=config['output_nonlinearity'],
    )

    # Load policy here

    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = PPOMAML(
        policy=policy,
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        learning_rate=config['learning_rate'],
        num_ppo_steps=config['num_ppo_steps'],
        num_minibatches=config['num_minibatches'],
        clip_eps=config['clip_eps'], 
        clip_outer=config['clip_outer'],
        target_outer_step=config['target_outer_step'],
        target_inner_step=config['target_inner_step'],
        init_outer_kl_penalty=config['init_outer_kl_penalty'],
        init_inner_kl_penalty=config['init_inner_kl_penalty'],
        adaptive_outer_kl_penalty=config['adaptive_outer_kl_penalty'],
        adaptive_inner_kl_penalty=config['adaptive_inner_kl_penalty'],
        anneal_factor=config['anneal_factor'],
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
    )

    trainer.train()

if __name__ == '__main__':

    sweep_params = {
        'algo': ['promp'],
        'seed': [1, 2, 3],

        'baseline': [LinearFeatureBaseline],

        'env': [BlueEnv],

        'rollouts_per_meta_task': [20],
        'max_path_length': [200],
        'parallel': [True],

        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],

        'hidden_sizes': [(64, 64)],
        'learn_std': [True],
        'hidden_nonlinearity': [tf.tanh],
        'output_nonlinearity': [None],

        'inner_lr': [0.001, 0.002, 0.01],
        'learning_rate': [1e-3],
        'num_ppo_steps': [5],
        'num_minibatches': [1],
        'clip_eps': [0.3],
        'clip_outer': [True],
        'target_outer_step': [0],
        'target_inner_step': [0.0],
        'init_outer_kl_penalty': [0],
        'init_inner_kl_penalty': [0, 5e-4],
        'adaptive_outer_kl_penalty': [False],
        'adaptive_inner_kl_penalty': [False],
        'anneal_factor': [1.0],

        'n_itr': [500],
        'meta_batch_size': [40],
        'num_inner_grad_steps': [1],
        'scope': [None],

        'exp_tag': ['v0']
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
