from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.meta_envs.mujoco.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_mb.meta_envs.rl2_env import rl2env
from meta_mb.algos.ppo import PPO
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.meta_samplers import MAMLSampler
from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
from meta_mb.policies.gaussian_rnn_policy import GaussianRNNPolicy
import os
from meta_mb.logger import logger
import json
import numpy as np

maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):
    baseline = LinearFeatureBaseline()
    env = rl2env(HalfCheetahRandDirecEnv())
    obs_dim = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape) + 1 + 1
    policy = GaussianRNNPolicy(
            name="meta-policy",
            obs_dim=obs_dim,
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
            cell_type=config['cell_type']
        )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=1,
    )

    sample_processor = RL2SampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = PPO(
        policy=policy,
        learning_rate=config['learning_rate'],
        max_epochs=config['max_epochs']
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
    )
    trainer.train()


if __name__=="__main__":
    idx = np.random.randint(0, 1000)
    data_path = maml_zoo_path + '/data/rl2/test_%d' % idx
    logger.configure(dir=data_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')
    config = json.load(open(maml_zoo_path + "/configs/rl2_config.json", 'r'))
    json.dump(config, open(data_path + '/params.json', 'w'))
    main(config)
