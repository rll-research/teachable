import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.meta_envs.mujoco.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_mb.meta_envs.mujoco.ant_rand_direc import AntRandDirecEnv
from meta_mb.meta_envs.mujoco.ant_rand_goal import AntRandGoalEnv
from meta_mb.meta_envs.mujoco.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_mb.meta_envs.mujoco.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.meta_algos.trpo_maml import TRPOMAML
from meta_mb.trainers.meta_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_mb.logger import logger

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'maml-kate-def'

def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last_gap', snapshot_gap=50)
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    with sess.as_default() as sess:

        # Instantiate classes
        set_seed(kwargs['seed'])

        baseline = kwargs['baseline']()

        env = normalize(kwargs['env']()) # Wrappers?

        policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=kwargs['meta_batch_size'],
            hidden_sizes=kwargs['hidden_sizes'],
            learn_std=kwargs['learn_std'],
            hidden_nonlinearity=kwargs['hidden_nonlinearity'],
            output_nonlinearity=kwargs['output_nonlinearity'],
        )

        # Load policy here

        sampler = MetaSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=kwargs['rollouts_per_meta_task'],
            meta_batch_size=kwargs['meta_batch_size'],
            max_path_length=kwargs['max_path_length'],
            parallel=kwargs['parallel'],
        )

        sample_processor = MAMLSampleProcessor(
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
            sampler=sampler,
            sample_processor=sample_processor,
            n_itr=kwargs['n_itr'],
            num_inner_grad_steps=kwargs['num_inner_grad_steps'],
            sess=sess,
        )

        trainer.train()


if __name__ == '__main__':    

    sweep_params = {
        'algo': ["MAML"],
        'seed' : [1, 2, 3],

        'baseline': [LinearFeatureBaseline],

        'env': [HumanoidRandDirec2DEnv],

        'rollouts_per_meta_task': [20],
        'max_path_length': [200],
        'parallel': [True],

        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],

        'hidden_sizes': [(128, 128)],
        'learn_std': [True],
        'hidden_nonlinearity': [tf.tanh],
        'output_nonlinearity': [None],

        'inner_lr': [0.1],
        'inner_type': ['log_likelihood'],
        'step_size': [0.01],
        'exploration': [False],

        'n_itr': [1001],
        'meta_batch_size': [40],
        'num_inner_grad_steps': [1],
        'scope': [None],

        'exp_tag': [''], # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
