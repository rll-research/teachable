from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv
from meta_mb.meta_algos.ppo_maml import PPOMAML
from meta_mb.trainers.meta_trainer import Trainer
from meta_mb.samplers.meta_samplers import MAMLSampler
from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
import os
from meta_mb.logger import logger
import json
import numpy as np

maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):
    baseline = LinearFeatureBaseline()
    # env = normalize(HalfCheetahRandDirecEnv())
    env = HopperRandParamsEnv(3.5)
    policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
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
        num_inner_grad_steps=config['num_inner_grad_steps'],  # This is repeated in MAMLPPO, it's confusing
    )
    trainer.train()

if __name__=="__main__":
    idx = np.random.randint(0, 1000)
    data_path = maml_zoo_path + '/data/ppo/test_%d' % idx
    logger.configure(dir=data_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')
    config = json.load(open(maml_zoo_path + "/configs/ppo_maml_config.json", 'r'))
    json.dump(config, open(data_path + '/params.json', 'w'))
    main(config)