import os
import json
import numpy as np
# from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.svg_1 import SVG1
from meta_mb.trainers.svg_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.metrpo_samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.dynamics.probabilistic_mlp_dynamics import ProbMLPDynamics
from meta_mb.baselines.nn_basline import NNValueFun
from meta_mb.logger import logger

from meta_mb.envs import mb_envs

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'svg'


def run_experiment(kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    # Instantiate classes
    set_seed(kwargs['seed'])

    env = normalize(kwargs['env']())  # Wrappers?

    baseline = NNValueFun('value-function',
                          env,
                          hidden_nonlinearity=kwargs['vfun_hidden_nonlinearity'],
                          hidden_sizes=kwargs['vfun_hidden_sizes'],
                          output_nonlinearity=kwargs['vfun_output_nonlinearity'],
                          learning_rate=kwargs['vfun_learning_rate'],
                          batch_size=kwargs['vfun_batch_size'],
                          buffer_size=kwargs['vfun_buffer_size'],
                          )

    policy = GaussianMLPPolicy(
        name="policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        hidden_sizes=kwargs['policy_hidden_sizes'],
        learn_std=kwargs['policy_learn_std'],
        output_nonlinearity=kwargs['policy_output_nonlinearity'],
    )

    dynamics_model = ProbMLPDynamics('prob-dynamics',
                                     env=env,
                                     hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                     hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                     output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                     learning_rate=kwargs['dynamics_learning_rate'],
                                     batch_size=kwargs['dynamics_batch_size'],
                                     buffer_size=kwargs['dynamics_buffer_size'],
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
        num_grad_steps=kwargs['num_rollouts'] * kwargs['max_path_length'] // kwargs['svg_batch_size'],
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
    )

    trainer.train()


def parse_env(env_name):
    if env_name == 'gym_acrobot':
        env = mb_envs.acrobot.AcrobotEnv
        length = 200
    elif env_name == 'gym_ant':
        env = mb_envs.ant.AntEnv
        length = 1000
    elif env_name == 'gym_cartpole':
        env = mb_envs.cartpole.CartPoleEnv
        length = 200
    elif env_name == 'gym_cheetah':
        env = mb_envs.half_cheetah.HalfCheetahEnv
        length = 1000
    elif env_name == 'gym_hopper':
        env = mb_envs.hopper.HopperEnv
        length = 1000
    elif env_name == 'gym_invertedPendulum':
        env = mb_envs.inverted_pendulum.InvertedPendulumEnv
        length = 100
    elif env_name == 'gym_mountain':
        env = mb_envs.mountain_car.Continuous_MountainCarEnv
        length = 200
    elif env_name == 'gym_pendulum':
        env = mb_envs.pendulum.PendulumEnv
        length = 200
    elif env_name == 'gym_reacher':
        env = mb_envs.reacher.ReacherEnv
        length = 50
    elif env_name == 'gym_swimmer':
        env = mb_envs.swimmer.SwimmerEnv
        length = 1000
    elif env_name == 'gym_walker2d':
        env = mb_envs.walker2d.Walker2dEnv
        length = 1000

    elif env_name == 'gym_humanoid':
        env = mb_envs.humanoid.HumanoidEnv
        length = 1000
    elif env_name == 'gym_slimhumanoid':
        env = mb_envs.slimhumanoid.SlimHumanoidEnv
        length = 1000
    elif env_name == 'gym_nostopslimhumanoid':
        env = mb_envs.nostopslimhumanoid.NoStopSlimHumanoidEnv
        length = 1000

    elif env_name == 'gym_pendulumO01':
        env = mb_envs.pendulumO01.PendulumO01Env
        length = 200
    elif env_name == 'gym_pendulumO001':
        env = mb_envs.pendulumO001.PendulumO001Env
        length = 200
    elif env_name == 'gym_cartpoleO01':
        env = mb_envs.cartpoleO01.CartPole01Env
        length = 200
    elif env_name == 'gym_cartpoleO001':
        env = mb_envs.cartpoleO001.CartPoleO001Env
        length = 200
    elif env_name == 'gym_cheetahA01':
        env = mb_envs.half_cheetahA01.HalfCheetahA01Env
        length = 1000
    elif env_name == 'gym_cheetahA003':
        env = mb_envs.half_cheetahA003.HalfCheetahA003Env
        length = 1000
    elif env_name == 'gym_cheetahO01':
        env = mb_envs.half_cheetahO01.HalfCheetahO01Env
        length = 1000
    elif env_name == 'gym_cheetahO001':
        env = mb_envs.half_cheetahO001.HalfCheetahO001Env
        length = 1000

    elif env_name == 'gym_fwalker2d':
        env = mb_envs.fwalker2d.FWalker2dEnv
        length = 1000
    elif env_name == 'gym_fswimmer':
        env = mb_envs.fswimmer.FSwimmerEnv
        length = 1000
    elif env_name == 'gym_fhopper':
        env = mb_envs.fhopper.FHopperEnv
        length = 1000
    elif env_name == 'gym_fant':
        env = mb_envs.fant.FAntEnv
        length = 1000

    return env, length


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='svg_mb_bm.')
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--exp_name_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_iteration', type=int, default=50)
    parser.add_argument('--svg_learning_rate', type=float, default=1e-4)
    parser.add_argument('--kl_penalty', type=float, default=1e-3)
    parser.add_argument('--svg_max_buffer_size', type=int, default=25000)
    args = parser.parse_args()

    env, env_length = parse_env(args.env_name)

    num_rollouts = int(4 * (1000 / env_length))  # fix 4000 timesteps
    assert 1000 % env_length == 0

    params = {
        'seed': args.seed,

        'algo': 'svg',
        'env': env,

        # Problem Conf
        'n_itr': args.num_iteration,
        'max_path_length': env_length,
        'discount': 0.99,
        'gae_lambda': 1.,
        'normalize_adv': True,
        'positive_adv': False,

        # Env Sampling
        'num_rollouts': num_rollouts,
        'n_parallel': 4,  # Parallelized across 4 cores (4 cores? 2 cores?)

        # Dynamics Model
        'dynamics_hidden_sizes': (500, 500),
        'dyanmics_hidden_nonlinearity': 'relu',
        'dyanmics_output_nonlinearity': None,
        'dynamics_max_epochs': 50,
        'dynamics_learning_rate': 1e-3,
        'dynamics_batch_size': 128,
        'dynamics_buffer_size': 25000,

        # Value Function
        'vfun_hidden_sizes': (400, 200),
        'vfun_hidden_nonlinearity': 'relu',
        'vfun_output_nonlinearity': None,
        'vfun_max_epochs': 50,
        'vfun_learning_rate': 5e-4,
        'vfun_batch_size': 32,
        'vfun_buffer_size': 10000,

        # Policy
        'policy_hidden_sizes': (100, 100),
        'policy_learn_std': True,
        'policy_output_nonlinearity': None,

        # Algo
        'svg_learning_rate': args.svg_learning_rate,
        'svg_batch_size': 64,
        'svg_max_buffer_size': args.svg_max_buffer_size,
        'kl_penalty': args.kl_penalty,

        # Misc
        'scope': None,
        'exp_tag': '',  # For changes besides hyperparams
        # Add time-stamp here to not overwrite the logging
        'exp_name': args.exp_name_prefix + args.env_name + \
                '_seed_' + str(args.seed)
    }

    run_experiment(params)
