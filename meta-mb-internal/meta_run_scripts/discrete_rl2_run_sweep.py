from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.meta_envs.point.point_env_1d import MetaPointEnv
from meta_mb.meta_envs.rl2_env import rl2env
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.ppo import PPO
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
from meta_mb.policies.discrete_rnn_policy import DiscreteRNNPolicy
import os
import shutil
from meta_mb.logger import logger
import json
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
import tensorflow as tf
from babyai.levels.iclr19_levels import *
from babyai.levels.curriculum import Curriculum
from babyai.oracle.post_action_advice import PostActionAdvice
from babyai.oracle.physical_correction import PhysicalCorrections
from babyai.oracle.landmark_correction import LandmarkCorrection
from babyai.oracle.demo_corrections import DemoCorrections

from babyai.bot import Bot
import joblib

INSTANCE_TYPE = 'c4.xlarge'
# PREFIX = 'V0curriculum'
PREFIX = 'debug22'
# PREFIX = 'WhyNotFollowTeacherYesEntropy'
# PREFIX = 'ArePreLevelsGood'
# PREFIX = 'DROPOUT'
# PREFIX = 'FOLLOWSTRICT'
# PREFIX = 'REALLYEASYnormallevel'
# PREFIX = 'BIGMODEL'
# PREFIX = 'SUPERVISEDDROPOUTMR'
# PREFIX = 'DISCOUNT.9'

def get_exp_name(config):
    EXP_NAME = PREFIX
    EXP_NAME += '_teacher' + str(config['feedback_type'])
    EXP_NAME += '_persist'
    if config['persist_goal']:
        EXP_NAME += "g"
    if config['persist_objs']:
        EXP_NAME += "o"
    if config['persist_agent']:
        EXP_NAME += "a"
    if config['pre_levels']:
        EXP_NAME += '_pre'
    EXP_NAME += '_droptype' + str(config['dropout_type'])
    EXP_NAME += '_dropinc' + str(config['dropout_incremental'])
    EXP_NAME += '_dropgoal' + str(config['dropout_goal'])
    EXP_NAME += '_ent' + str(config['entropy_bonus'])
    EXP_NAME += '_lr' + str(config['learning_rate'])
    EXP_NAME += 'corr' + str(config['dropout_correction'])
    EXP_NAME += '_currfn' + config['advance_curriculum_func']
    print("EXPERIMENT NAME:", EXP_NAME)
    return EXP_NAME

def run_experiment(**config):
    set_seed(config['seed'])
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    original_saved_path = config['saved_path']
    with sess.as_default() as sess:
        arguments = {
            "start_loc": 'all',
            "include_holdout_obj": False,
            "persist_goal": config['persist_goal'],
            "persist_objs": config['persist_objs'],
            "persist_agent": config['persist_agent'],
            "dropout_goal": config['dropout_goal'],
            "dropout_correction": config['dropout_correction'],
            "dropout_independently": config['dropout_independently'],
            "dropout_type": config['dropout_type'],
            "dropout_incremental": None if config['dropout_incremental'] is None else config['dropout_incremental'][1],
            "feedback_type": config["feedback_type"],
            "feedback_always": config["feedback_always"],
            "num_meta_tasks": config["rollouts_per_meta_task"],
        }
        if config['saved_path'] is not None:
            saved_model = joblib.load(config['saved_path'])
            if 'config' in saved_model:
                if not config['override_old_config']:
                    config = saved_model['config']
            set_seed(config['seed'])
            policy = saved_model['policy']
            baseline = saved_model['baseline']
            curriculum_step = saved_model['curriculum_step']
            env = rl2env(normalize(Curriculum(config['advance_curriculum_func'], start_index=curriculum_step,
                                              **arguments)),
                         ceil_reward=config['ceil_reward'])
            start_itr = saved_model['itr']
            reward_predictor = saved_model['reward_predictor']
            if 'supervised_model' in saved_model:
                supervised_model = saved_model['supervised_model']
            else:
                supervised_model = None

        else:
            baseline = config['baseline']()
            env = rl2env(normalize(Curriculum(config['advance_curriculum_func'],
                                              pre_levels=config['pre_levels'], **arguments)),
                         ceil_reward=config['ceil_reward'])
            obs_dim = env.reset().shape[0]
            policy = DiscreteRNNPolicy(
                    name="meta-policy",
                    action_dim=np.prod(env.action_space.n),
                    obs_dim=obs_dim,
                    meta_batch_size=config['meta_batch_size'],
                    hidden_sizes=config['hidden_sizes'],
                    cell_type=config['cell_type']
                )
            reward_predictor = DiscreteRNNPolicy(
                name="reward-predictor",
                action_dim=2,
                obs_dim=obs_dim - 1,
                meta_batch_size=config['meta_batch_size'],
                hidden_sizes=config['hidden_sizes'],
                cell_type=config['cell_type']
            )
            if config['il_comparison']:
                supervised_model = DiscreteRNNPolicy(
                    name="supervised-policy",
                    action_dim=np.prod(env.action_space.n),
                    obs_dim=obs_dim,
                    meta_batch_size=config['meta_batch_size'],
                    hidden_sizes=config['hidden_sizes'],
                    cell_type=config['cell_type'],
                    preprocess_obs_type='meta_rollout_dropout',#'full_dropout',
                )
            else:
                supervised_model = None
            start_itr = 0
            curriculum_step = env.index

        sampler = MetaSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=config['rollouts_per_meta_task'],
            meta_batch_size=config['meta_batch_size'],
            max_path_length=config['max_path_length'],
            parallel=config['parallel'],
            envs_per_task=1,
            reward_predictor=reward_predictor,
            supervised_model=supervised_model,
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
            supervised_model=supervised_model,
            learning_rate=config['learning_rate'],
            max_epochs=config['max_epochs'],
            backprop_steps=config['backprop_steps'],
            reward_predictor=reward_predictor,
            entropy_bonus=config['entropy_bonus'],
        )

        EXP_NAME = get_exp_name(config)
        exp_dir = os.getcwd() + '/data/' + EXP_NAME + "_" + str(config['seed'])
        if original_saved_path is None:
            if os.path.isdir(exp_dir):
                shutil.rmtree(exp_dir)
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'], snapshot_mode='level', # TODO: change back to level
                         snapshot_gap=50, step=start_itr)
        json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=deepcopy(env),
            sampler=sampler,
            sample_processor=sample_processor,
            n_itr=config['n_itr'],
            sess=sess,
            start_itr=start_itr,
            reward_threshold=config['reward_threshold'],
            exp_name=exp_dir,
            curriculum_step=curriculum_step,
            config=config,
            increase_dropout_threshold=0 if config['dropout_incremental'] is None else config['dropout_incremental'][0]
        )
        trainer.train()

set
if __name__ == '__main__':
    base_path = '/home/olivia/Documents/Teachable/babyai/meta-mb-internal/data/'
    sweep_params = {
        'saved_path': [None],#base_path + 'SUPERVISED_teacherPreActionAdvice_persistgoa_droptypestep_dropgoal0_ent0.001_lr0.01corr0_currfnsmooth_4/latest.pkl'],
        'override_old_config': [False],  # only relevant when restarting a run; do we use the old config or the new?
        'persist_goal': [True],
        'persist_objs': [True],
        'persist_agent': [True],
        'dropout_goal': [0],
        'dropout_correction': [0],
        'dropout_type': ['step'], # Options are [step, rollout, meta_rollout, meta_rollout_start]
        'dropout_incremental': [(0.8, 0.2)], # Options are None or (threshold, increment), where threshold is the accuracy level at which you increase the amount of dropout,
                                   # and increment is the proportion of the total dropout rate which gets added each time # TODO: change back
        'dropout_independently': [True],  # Don't ensure we have at least one source of feedback
        'reward_threshold': [.95],
        "feedback_type": ["PreActionAdvice"],  # Options are [None, "PreActionAdvice", "PostActionAdvice"]
        "rollouts_per_meta_task": [2],
        'ceil_reward': [True],
        'advance_curriculum_func': ['smooth'],
        'entropy_bonus': [1e-3],
        'feedback_always': [True],
        'pre_levels': [False],
        'il_comparison': [True],

        'algo': ['rl2'],
        'seed': [4],
        'baseline': [LinearFeatureBaseline],
        'env': [MetaPointEnv],
        'meta_batch_size': [100],
        "hidden_sizes": [(64,), (128,)],#[(256,), (256,), (256,)],#
        'backprop_steps': [50, 100, 200],
        "parallel": [False], # TODO: consider changing this back! I think parallel has been crashing my computer.
        "max_path_length": [float('inf')],  # Dummy; we don't time out episodes (they time out by themselves)
        "discount": [0.95],
        "gae_lambda": [1.0],
        "normalize_adv": [True],
        "positive_adv": [False],
        "learning_rate": [1e-2],
        "max_epochs": [5],
        "cell_type": ["lstm"],
        "num_minibatches": [1],
        "n_itr": [10000],
        'exp_tag': ['v0'],
        'log_rand': [0, 1, 2, 3],
    }
    run_sweep(run_experiment, sweep_params, PREFIX, INSTANCE_TYPE)
