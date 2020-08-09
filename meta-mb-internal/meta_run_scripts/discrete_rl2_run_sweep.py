from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.meta_envs.rl2_env import rl2env
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.ppo import PPO
from meta_mb.algos.ppo_torch import PPOAlgo
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
from meta_mb.policies.discrete_rnn_policy import DiscreteRNNPolicy
from meta_mb.policies.gaussian_rnn_policy import GaussianRNNPolicy
from babyai.model import ACModel
from meta_mb.trainers.il_trainer import ImitationLearning
from babyai.arguments import ArgumentParser

import os
import shutil
from meta_mb.logger import logger
import json
import numpy as np
from gym import spaces
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
PREFIX = 'FIXEDITHINK'


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
    if config['il_comparison']:
        EXP_NAME += '_IL'
    if config['self_distill']:
        EXP_NAME += '_SD'
    if config['intermediate_reward']:
        EXP_NAME += '_dense'
    EXP_NAME += '_droptype' + str(config['dropout_type'])
    EXP_NAME += '_dropinc' + str(config['dropout_incremental'])
    EXP_NAME += '_dropgoal' + str(config['dropout_goal'])
    EXP_NAME += '_disc' + str(config['discount'])
    EXP_NAME += '_thresh' + str(config['success_threshold'])
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
        if original_saved_path is not None:
            saved_model = joblib.load(config['saved_path'])
            if 'config' in saved_model:
                if not config['override_old_config']:
                    config = saved_model['config']
                    config['intermediate_reward'] = False  # TODO
                    config['reward_predictor_type'] = 'discrete'
                    config['grad_clip_threshold'] = None
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
            "feedback_type": config["feedback_type"],
            "feedback_always": config["feedback_always"],
            "num_meta_tasks": config["rollouts_per_meta_task"],
            "intermediate_reward": config["intermediate_reward"]
        }
        if original_saved_path is not None:
            set_seed(config['seed'])
            policy = saved_model['policy']
            policy.hidden_state = None
            baseline = saved_model['baseline']
            curriculum_step = config['level']
            env = rl2env(normalize(Curriculum(config['advance_curriculum_func'], start_index=curriculum_step,
                                              **arguments)),
                         ceil_reward=config['ceil_reward'])
            start_itr = saved_model['itr']
            reward_predictor = saved_model['reward_predictor']
            reward_predictor.hidden_state = None
            if 'supervised_model' in saved_model:
                supervised_model = saved_model['supervised_model']
            else:
                supervised_model = None

        else:
            baseline = config['baseline']()
            env = rl2env(normalize(Curriculum(config['advance_curriculum_func'], start_index=config['level'],
                                              pre_levels=config['pre_levels'], **arguments)),
                         ceil_reward=config['ceil_reward'])
            obs_dim = env.reset().shape[0]
            image_dim = 128
            memory_dim = config['memory_dim']
            instr_dim = config['instr_dim']
            use_instr = True
            instr_arch = 'bigru'
            use_mem = True
            arch = 'bow_endpool_res'
            advice_start_index = 160
            advice_end_index = advice_start_index + env.action_space.n + 1
            policy = ACModel(obs_space=obs_dim,
                             action_space=env.action_space,
                             env=env,
                             image_dim=image_dim,
                             memory_dim=memory_dim,
                             instr_dim=instr_dim,
                             lang_model=instr_arch,
                             use_instr=use_instr,
                             use_memory=use_mem,
                             arch=arch,
                             advice_dim=128,
                             advice_start_index=advice_start_index,
                             advice_end_index=advice_end_index)


            reward_predictor = ACModel(obs_space=obs_dim - 1,  # TODO: change into Discrete(3) and do 3-way classification
                                     action_space=spaces.Discrete(2),
                                     env=env,
                                     image_dim=image_dim,
                                     memory_dim=memory_dim,
                                     instr_dim=instr_dim,
                                     lang_model=instr_arch,
                                     use_instr=use_instr,
                                     use_memory=use_mem,
                                     arch=arch,
                                     advice_dim=128,
                                     advice_start_index=advice_start_index,
                                     advice_end_index=advice_end_index)
            assert not (config['il_comparison'] and config['self_distill'])
            if config['il_comparison']:
                obs_dim = env.reset().shape[0]
                image_dim = 128
                memory_dim = config['memory_dim']
                instr_dim = config['instr_dim']
                use_instr = True
                instr_arch = 'bigru'
                use_mem = True
                arch = 'bow_endpool_res'
                advice_start_index = 160
                advice_end_index = advice_start_index + env.action_space.n + 1
                supervised_model = ACModel(obs_space=obs_dim,
                                         action_space=env.action_space,
                                         env=env,
                                         image_dim=image_dim,
                                         memory_dim=memory_dim,
                                         instr_dim=instr_dim,
                                         lang_model=instr_arch,
                                         use_instr=use_instr,
                                         use_memory=use_mem,
                                         arch=arch,
                                         advice_dim=128,
                                         advice_start_index=advice_start_index,
                                         advice_end_index=advice_end_index)
            elif config['self_distill']:
                supervised_model = policy
            else:
                supervised_model = None
            start_itr = 0
            curriculum_step = env.index

        parser = ArgumentParser()
        args = parser.parse_args()
        args.model = 'default_il'
        args.recurrence = config['backprop_steps']
        il_trainer = ImitationLearning(supervised_model, env, args, config['distill_with_teacher'])
        rp_trainer = ImitationLearning(reward_predictor, env, args, distill_with_teacher=True, reward_predictor=True)

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

        algo = PPOAlgo(policy, args.frames_per_proc, config['discount'], args.lr, args.beta1, args.beta2,
                       config['gae_lambda'],
                       args.entropy_coef, .5, .5, args.recurrence,
                       args.optim_eps, .2, 4, config['meta_batch_size'])

        EXP_NAME = get_exp_name(config)
        exp_dir = os.getcwd() + '/data/' + EXP_NAME + "_" + str(config['seed'])
        if original_saved_path is None:
            if os.path.isdir(exp_dir):
                shutil.rmtree(exp_dir)
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'], snapshot_mode='level',
                         snapshot_gap=50, step=start_itr)
        json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

        action_dim = env.action_space.n + 1

        if config['distill_with_teacher']:  # TODO: generalize this for multiple feedback types
            teacher_info = []
        else:
            null_val = np.zeros(action_dim)
            start_index = 160
            null_val[-1] = 1
            teacher_info = [{"indices": np.arange(start_index, start_index + action_dim), "null": null_val}]

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=deepcopy(env),
            sampler=sampler,
            sample_processor=sample_processor,
            n_itr=config['n_itr'],
            sess=sess,
            start_itr=start_itr,
            success_threshold=config['success_threshold'],
            accuracy_threshold=config['accuracy_threshold'],
            exp_name=exp_dir,
            curriculum_step=curriculum_step,
            config=config,
            increase_dropout_threshold=float('inf') if config['dropout_incremental'] is None else
            config['dropout_incremental'][0],
            increase_dropout_increment=None if config['dropout_incremental'] is None else config['dropout_incremental'][
                1],
            advance_without_teacher=True,
            teacher_info=teacher_info,
            sparse_rewards=not config['intermediate_reward'],
            distill_only=config['distill_only'],
            il_trainer=il_trainer,
            source=config['source'],
            batch_size=config['meta_batch_size'],
            distill_with_teacher=config['distill_with_teacher'],
            supervised_model=supervised_model,
            reward_predictor=reward_predictor,
            rp_trainer=rp_trainer,
        )
        trainer.train()


if __name__ == '__main__':
    DEBUG = False  # Make this true to run a really quick run designed to sanity check the code runs
    base_path = 'data/'
    sweep_params = {
        'level': [4],
        "n_itr": [10000],
        'source': ['teacher'],  # options are agent or teacher (do we distill from the agent or the teacher?)
        'distill_with_teacher': [False],

        # Saving/loading/finetuning
        'saved_path': [None],
        # base_path + 'THRESHOLD++_teacherPreActionAdvice_persistgoa_droptypestep_dropinc(0.8, 0.2)_dropgoal0_disc0.9_thresh0.95_ent0.001_lr0.01corr0_currfnsmooth_4/latest.pkl'],#base_path + 'JUSTSUPLEARNINGL13distillation_batches10_4/latest.pkl'],
        'override_old_config': [True],  # only relevant when restarting a run; do we use the old config or the new?
        'distill_only': [False],

        # Meta
        'persist_goal': [True],
        'persist_objs': [True],
        'persist_agent': [True],
        "rollouts_per_meta_task": [1],  # TODO: change this back to > 1

        # Dropout
        'dropout_goal': [0],

        'dropout_correction': [0],
        'dropout_type': ['step'],  # Options are [step, rollout, meta_rollout, meta_rollout_start]
        'dropout_incremental': [None],
        # [(0.8, 0.2)], # Options are None or (threshold, increment), where threshold is the accuracy level at which you increase the amount of dropout,
        # and increment is the proportion of the total dropout rate which gets added each time
        'dropout_independently': [True],  # Don't ensure we have at least one source of feedback

        # Teacher
        "feedback_type": ["PreActionAdvice"],
        # Options are [None, "PreActionAdvice", "PostActionAdvice", "CartesianCorrections", "SubgoalCorrections"]
        'feedback_always': [True],

        # Curriculum
        'advance_curriculum_func': ['one_hot'],
        'pre_levels': [False],

        # Model/Optimization
        'entropy_bonus': [1e-2],  # 1e-2
        'grad_clip_threshold': [None],  # TODO: ask A about this:  grad goes from 10 to 60k.  Normal?
        "learning_rate": [1e-3],
        "hidden_sizes": [(512, 512)],
        "memory_dim": [1024],  # 2048
        "instr_dim": [128],  # 256
        "discount": [0.95],

        # Reward
        'reward_predictor_type': ['gaussian'],
        'intermediate_reward': [False],  # This turns the intermediate rewards on or off
        'success_threshold': [.95],
        'accuracy_threshold': [.9],
        'ceil_reward': [False],

        # Distillation
        'il_comparison': [True],  # 'full_dropout',#'meta_rollout_dropout',#'no_dropout'
        'self_distill': [False],

        # Arguments we basically never change
        'algo': ['rl2'],
        'seed': [4],
        'baseline': [LinearFeatureBaseline],
        'meta_batch_size': [20],
        'backprop_steps': [20],
        "parallel": [True],
        "max_path_length": [float('inf')],  # Dummy; we don't time out episodes (they time out by themselves)
        "gae_lambda": [1.0],
        "normalize_adv": [True],
        "positive_adv": [False],
        "max_epochs": [5],
        "cell_type": ["lstm"],
    }

    # DEBUG HPARAMS
    if DEBUG:
        sweep_params['meta_batch_size'] = [2]
        sweep_params['success_threshold'] = [0]
        sweep_params['accuracy_threshold'] = [0]
        sweep_params['hidden_sizes'] = [(2,)]
        sweep_params['backprop_steps'] = [1]
        sweep_params['max_path_length'] = [3]
        sweep_params['parallel'] = [False]
        sweep_params["memory_dim"] = [3]  # 2048
        sweep_params["instr_dim"] = [4]  # 256

    run_sweep(run_experiment, sweep_params, PREFIX, INSTANCE_TYPE)
