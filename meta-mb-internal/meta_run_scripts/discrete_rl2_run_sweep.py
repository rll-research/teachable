from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.meta_envs.rl2_env import rl2env
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.ppo_torch import PPOAlgo
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
from babyai.model import ACModel
from meta_mb.trainers.il_trainer import ImitationLearning
from babyai.arguments import ArgumentParser

import torch
import copy
import shutil
from meta_mb.logger import logger
import json
from gym import spaces
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from babyai.levels.iclr19_levels import *
from babyai.levels.curriculum import Curriculum

import joblib
import argparse
import pathlib

INSTANCE_TYPE = 'c4.xlarge'

def args_type(default):
  if isinstance(default, bool):
    return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int):
    return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  if isinstance(default, pathlib.Path):
    return lambda x: pathlib.Path(x).expanduser()
  if default is None:
    return str
  return type(default)


def get_exp_name(config):
    EXP_NAME = config['prefix']
    EXP_NAME += '_teacher' + str(config['feedback_type'])
    if config['distill_same_model']:
        EXP_NAME += '_SAME'
    if config['self_distill']:
        EXP_NAME += '_SD'
    if config['intermediate_reward']:
        EXP_NAME += '_dense'
    EXP_NAME += '_threshS' + str(config['success_threshold'])
    EXP_NAME += '_threshA' + str(config['accuracy_threshold'])
    EXP_NAME += '_lr' + str(config['learning_rate'])
    EXP_NAME += '_ent' + str(config['entropy_bonus'])
    EXP_NAME += '_currfn' + config['advance_curriculum_func']
    print("EXPERIMENT NAME:", EXP_NAME)
    return EXP_NAME

def get_advice_index(advice_start_index, config, env):
    advice_dim = 128
    if config['feedback_type'] == 'PreActionAdvice':
        advice_end_index = advice_start_index + env.action_space.n + 1
    elif config['feedback_type'] == 'CartesianCorrections':
        advice_end_index = advice_start_index + 160
    elif config['feedback_type'] == 'SubgoalCorrections':
        advice_end_index = advice_start_index + 17
    elif config['feedback_type'] is None:
        advice_end_index = 160
        advice_dim = 0
    else:
        raise NotImplementedError(config['feedback_type'])
    return advice_end_index, advice_dim


def run_experiment(**config):
    set_seed(config['seed'])
    original_saved_path = config['saved_path']
    if original_saved_path is not None:
        saved_model = joblib.load(config['saved_path'])
        if 'config' in saved_model:
            if not config['override_old_config']:
                config = saved_model['config']
    arguments = {
        "start_loc": 'all',
        "include_holdout_obj": False,
        "persist_goal": config['persist_goal'],
        "persist_objs": config['persist_objs'],
        "persist_agent": config['persist_agent'],
        "feedback_type": config["feedback_type"],
        "feedback_always": config["feedback_always"],
        "feedback_freq": config["feedback_freq"],
        "num_meta_tasks": config["rollouts_per_meta_task"],
        "intermediate_reward": config["intermediate_reward"]
    }
    advice_start_index = 160
    if original_saved_path is not None:
        set_seed(config['seed'])
        policy = saved_model['policy']
        optimizer = saved_model['optimizer']
        policy.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO: is this necessary?
        policy.hidden_state = None
        baseline = saved_model['baseline']
        curriculum_step = saved_model['curriculum_step']
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
        optimizer = None
        baseline = None
        env = rl2env(normalize(Curriculum(config['advance_curriculum_func'], start_index=config['level'], **arguments)),
                     ceil_reward=config['ceil_reward'])
        obs_dim = env.reset().shape[0]
        image_dim = 128
        memory_dim = config['memory_dim']
        instr_dim = config['instr_dim']
        use_instr = True
        instr_arch = 'bigru'
        use_mem = True
        arch = 'bow_endpool_res'
        advice_end_index, advice_dim = get_advice_index(advice_start_index, config, env)
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
                         advice_dim=advice_dim,
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
                                 advice_dim=advice_dim,
                                 advice_start_index=advice_start_index,
                                 advice_end_index=advice_end_index)
        if config['self_distill'] and not config['distill_same_model']:
            obs_dim = env.reset().shape[0]
            image_dim = 128
            memory_dim = config['memory_dim']
            instr_dim = config['instr_dim']
            use_instr = True
            instr_arch = 'bigru'
            use_mem = True
            arch = 'bow_endpool_res'
            supervised_model = ACModel(obs_space=obs_dim - 1,
                                     action_space=env.action_space,
                                     env=env,
                                     image_dim=image_dim,
                                     memory_dim=memory_dim,
                                     instr_dim=instr_dim,
                                     lang_model=instr_arch,
                                     use_instr=use_instr,
                                     use_memory=use_mem,
                                     arch=arch,
                                     advice_dim=advice_dim,
                                     advice_start_index=advice_start_index,
                                     advice_end_index=advice_end_index)
        elif config['self_distill']:
            supervised_model = policy
        else:
            supervised_model = None
        start_itr = 0
        curriculum_step = env.index
    parser = ArgumentParser()
    args = parser.parse_args([])
    args.entropy_coef = config['entropy_bonus']
    args.model = 'default_il'
    args.lr = config['learning_rate']
    args.recurrence = config['backprop_steps']
    args.clip_eps = config['clip_eps']
    if supervised_model is not None:
        il_trainer = ImitationLearning(supervised_model, env, args, distill_with_teacher=config['distill_with_teacher'])
    else:
        il_trainer = None
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

    envs = [copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            copy.deepcopy(env),
            ]
    algo = PPOAlgo(policy, envs, config['frames_per_proc'], config['discount'], args.lr, args.beta1, args.beta2,
                   config['gae_lambda'],
                   args.entropy_coef, config['value_loss_coef'], config['max_grad_norm'], args.recurrence,
                   args.optim_eps, config['clip_eps'], config['epochs'], config['meta_batch_size'])

    if optimizer is not None:
        algo.optimizer.load_state_dict(optimizer)

    EXP_NAME = get_exp_name(config)
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + "_" + str(config['seed'])
    if original_saved_path is None:
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
    log_formats = ['stdout', 'log', 'csv']
    is_debug = config['prefix'] == 'DEBUG'

    if not is_debug:
        log_formats.append('tensorboard')
        log_formats.append('wandb')
    logger.configure(dir=exp_dir, format_strs=log_formats,
                     snapshot_mode=config['save_option'],
                     snapshot_gap=50, step=start_itr, name=config['prefix'] + str(config['seed']), config=config, description=config['description'])
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    advice_end_index, advice_dim = get_advice_index(advice_start_index, config, env)
    if config['distill_with_teacher']:  # TODO: generalize this for multiple feedback types at once!
        teacher_info = []
    else:
        null_val = np.zeros(advice_end_index - advice_start_index)
        if len(null_val) > 0:
            null_val[-1] = 1
        teacher_info = [{"indices": np.arange(advice_start_index, advice_end_index), "null": null_val}]

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=deepcopy(env),
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        start_itr=start_itr,
        success_threshold=config['success_threshold'],
        accuracy_threshold=config['accuracy_threshold'],
        exp_name=exp_dir,
        curriculum_step=curriculum_step,
        config=config,
        advance_without_teacher=True,
        teacher_info=teacher_info,
        sparse_rewards=not config['intermediate_reward'],
        distill_only=config['distill_only'],
        il_trainer=il_trainer,
        source=config['source'],
        batch_size=config['meta_batch_size'],
        train_with_teacher=config['feedback_type'] is not None,
        distill_with_teacher=config['distill_with_teacher'],
        supervised_model=supervised_model,
        reward_predictor=reward_predictor,
        rp_trainer=rp_trainer,
        advance_levels=config['advance_levels'],
        is_debug=is_debug,
    )
    trainer.train()


if __name__ == '__main__':
    DEBUG = False  # Make this true to run a really quick run designed to sanity check the code runs
    base_path = 'data/'
    sweep_params = {
        'level': [0],
        "n_itr": [10000],
        'source': ['agent'],  # options are agent or teacher (do we distill from the agent or the teacher?)
        'distill_with_teacher': [False],
        'advance_levels': [True],  # can we advance levels, or do we have to stay on the current level?

        # Saving/loading/finetuning
        'prefix': ['DEBUG'],
        'description': ['yolo'],
        'saved_path': [None],
        'override_old_config': [False],  # only relevant when restarting a run; do we use the old config or the new?
        'distill_only': [False],
        'save_option': ['level'],  # options include 'all', 'level', 'latest', 'none', 'gap'

        # Meta
        'persist_goal': [True],
        'persist_objs': [True],
        'persist_agent': [True],
        "rollouts_per_meta_task": [1],  # TODO: change this back to > 1

        # Teacher
        "feedback_type": ["PreActionAdvice"],
        # Options are [None, "PreActionAdvice", "CartesianCorrections", "SubgoalCorrections"]
        'feedback_always': [False],
        'feedback_freq': [1],

        # Curriculum
        'advance_curriculum_func': ['one_hot'],  # TODO: double success doesn't get messed up when we use smooth

        # Model/Optimization
        'entropy_bonus': [.0005],  # TODO: .01 default
        'grad_clip_threshold': [1],  # TODO: not being used any more
        "learning_rate": [1e-3],  # TODO: 1e-4 default
        "memory_dim": [512],  #1024, 2048
        "instr_dim": [64],  #128, 256
        "discount": [0.9],

        "value_loss_coef": [.05],  # .5 is default
        "max_grad_norm": [.5],  # .5 is default
        "clip_eps": [.2],  # .2 is default
        "epochs": [4],  # 4 is default
        "frames_per_proc": [40],

        # Reward
        'intermediate_reward': [True],  # This turns the intermediate rewards on or off
        'success_threshold': [.95],
        'accuracy_threshold': [.9],
        'ceil_reward': [False],  # TODO: is this still being used?

        # Distillations
        'self_distill': [False],
        'distill_same_model': [False],  # 'full_dropout',#'meta_rollout_dropout',#'no_dropout'

        # Arguments we basically never change
        'seed': [2],
        'baseline': [None],
        'meta_batch_size': [20],  # TODO: big is default
        'backprop_steps': [20],  # In the babyai paper, they use 20 for the small model, 80 for the big model
        "parallel": [True],
        "max_path_length": [float('inf')],  # Dummy; we don't time out episodes (they time out by themselves)
        "gae_lambda": [.99],
        "normalize_adv": [True],
        "positive_adv": [False],

    }
    DEFAULT = 'DEFAULT'
    parser = argparse.ArgumentParser()
    for key, value in sweep_params.items():
        parser.add_argument(f'--{key}', default=DEFAULT)
    args = parser.parse_args()
    for k in vars(args):
        v = getattr(args, k)
        if not v == DEFAULT:
            arg_type = args_type(sweep_params[k][0])
            sweep_params[k] = [arg_type(v)]

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

    run_sweep(run_experiment, sweep_params, sweep_params['prefix'][0], parser, INSTANCE_TYPE)
