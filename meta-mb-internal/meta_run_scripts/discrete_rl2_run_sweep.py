from meta_mb.meta_envs.rl2_env import rl2env
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.ppo_torch import PPOAlgo
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
from babyai.model import ACModel, Reconstructor
from meta_mb.trainers.il_trainer import ImitationLearning
from babyai.arguments import ArgumentParser
from babyai.utils.obs_preprocessor import make_obs_preprocessor, make_obs_preprocessor_choose_teachers
from babyai.teacher_schedule import make_teacher_schedule
from babyai.levels.augment import DataAugmenter
from scripts.test_generalization import make_log_fn

import torch
import copy
import shutil
from meta_mb.logger import logger
from gym import spaces
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from babyai.levels.iclr19_levels import *
from babyai.levels.curriculum import Curriculum
import pathlib
import joblib


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


def get_exp_name(args):
    EXP_NAME = args.prefix
    return EXP_NAME
    feedback_type = str(args.feedback_type)
    feedback_type = ''.join([char for char in feedback_type[1:-1] if not char in ["'", "[", "]", ",", " "]])
    EXP_NAME += '_teacher' + feedback_type
    # if args.distill_same_model:
    #     EXP_NAME += '_SAME'
    if args.self_distill:
        EXP_NAME += '_SD'
    # if args.intermediate_reward:
    #     EXP_NAME += '_dense'
    EXP_NAME += '_threshS' + str(args.success_threshold_rl)
    EXP_NAME += '_threshAR' + str(args.accuracy_threshold_rl)
    EXP_NAME += '_threshAD' + str(args.accuracy_threshold_distill_teacher)
    EXP_NAME += '_lr' + str(args.lr)
    EXP_NAME += '_ent' + str(args.entropy_coef)
    # EXP_NAME += '_currfn' + args.advance_curriculum_func
    print("EXPERIMENT NAME:", EXP_NAME)
    return EXP_NAME


def load_model(args):
    original_config = args
    saved_model = joblib.load(args.saved_path)
    if 'args' in saved_model:
        if not args.override_old_config:
            args = saved_model['args']
    set_seed(args.seed)
    policy_dict = saved_model['policy']
    optimizer = saved_model.get('optimizer', None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for policy in policy_dict.values():
        policy.device = device
    if original_config.continue_train is True:
        start_itr = saved_model['itr']
        curriculum_step = saved_model['curriculum_step']
    else:
        start_itr = 0
        curriculum_step = args.level
    il_optimizer = saved_model.get('il_optimizer', None)
    log_dict = saved_model.get('log_dict', {})
    return policy_dict, optimizer, start_itr, curriculum_step, args, \
        il_optimizer, log_dict


def run_experiment(**config):
    parser = ArgumentParser()
    args = parser.parse_args()
    for k, v in config.items():
        setattr(args, k, v)

    set_seed(args.seed)
    original_saved_path = args.saved_path
    if original_saved_path is not None:
        policy_dict, optimizer, start_itr, curriculum_step, args, \
            il_optimizer, log_dict = load_model(args)
    else:
        il_optimizer = None
        log_dict = {}

    if args.env in ['point_mass', 'ant']:
        args.no_instr = True
        args.discrete = False
    elif args.env in ['babyai']:
        args.discrete = True
    else:
        raise NotImplementedError(f'Unknown env {args.env}')
    args.discrete = args.discrete

    arguments = {
        "start_loc": 'all',
        "include_holdout_obj": not args.leave_out_object,
        "persist_goal": not args.reset_goal,
        "persist_objs": not args.reset_objs,
        "persist_agent": not args.reset_agent,
        "feedback_type": args.feedback_type,
        "feedback_freq": args.feedback_freq,
        "cartesian_steps": args.cartesian_steps,
        "num_meta_tasks": args.rollouts_per_meta_task,
        "intermediate_reward": args.reward_type == 'dense',
        "reward_type": args.reward_type,
        "fully_observed": args.fully_observed,
        "padding": args.padding,
        "args": args,
    }
    teacher_schedule = make_teacher_schedule(args.feedback_type, args.teacher_schedule,
                                             args.success_intervention_cutoff,
                                             args.accuracy_intervention_cutoff)
    teacher_train_dict, _ = teacher_schedule(0, 0, 0)
    if args.zero_all_thresholds:
        args.success_threshold_rl = 0
        args.success_threshold_rollout_teacher = 0
        args.success_threshold_rollout_no_teacher = 0
        args.accuracy_threshold_rl = 0
        args.accuracy_threshold_distill_teacher = 0
        args.accuracy_threshold_distill_no_teacher = 0
        args.accuracy_threshold_rollout_teacher = 0
        args.accuracy_threshold_rollout_no_teacher = 0
    if original_saved_path is not None:
        env = rl2env(normalize(Curriculum(args.advance_curriculum_func, env=args.env, start_index=curriculum_step,
                                          curriculum_type=args.curriculum_type, **arguments),
                               normalize_actions=args.act_norm, normalize_reward=args.rew_norm,
                               ), ceil_reward=args.ceil_reward)
        try:
            teacher_null_dict = env.teacher.null_feedback()
        except Exception as e:
            teacher_null_dict = {}
        include_zeros = args.include_zeros or args.same_model
        obs_preprocessor = make_obs_preprocessor(teacher_null_dict, include_zeros=include_zeros)
        teachers_list = list(teacher_null_dict.keys()) + ['none']
    else:
        optimizer = None
        env = rl2env(normalize(Curriculum(args.advance_curriculum_func, env=args.env, start_index=args.level,
                                          curriculum_type=args.curriculum_type,
                                          **arguments), normalize_actions=args.act_norm, normalize_reward=args.rew_norm)
                     , ceil_reward=args.ceil_reward)
        obs = env.reset()
        args.advice_size = sum([np.prod(obs[k].shape) for k in teacher_train_dict.keys() if k in obs])
        if args.no_teacher:
            args.advice_size = 0

        try:
            teacher_null_dict = env.teacher.null_feedback()
        except Exception as e:
            teacher_null_dict = {}
        include_zeros = args.include_zeros or args.same_model
        obs_preprocessor = make_obs_preprocessor(teacher_null_dict, include_zeros=include_zeros)

        policy_dict = {}
        args.reconstruct_advice_size = sum([np.prod(obs[teacher].shape) for teacher in teacher_null_dict.keys() if teacher in obs])
        teachers_list = list(teacher_null_dict.keys()) + ['none']
        for teacher in teachers_list:
            if not args.include_zeros and not args.same_model:
                args.advice_size = 0 if teacher == 'none' else np.prod(obs[teacher].shape)
            if args.same_model and not teacher == teachers_list[0]:
                policy = policy_dict[teachers_list[0]]
            else:
                policy = ACModel(action_space=env.action_space,
                                 env=env,
                                 args=args)
            policy_dict[teacher] = policy

        start_itr = 0
        curriculum_step = env.index

    args.model = 'default_il'
    if args.reconstruction:
        reconstructor_dict = {k: Reconstructor(env, args) for k in teachers_list}
    else:
        reconstructor_dict = None
    il_trainer = ImitationLearning(policy_dict, env, args, distill_with_teacher=False,
                                   preprocess_obs=obs_preprocessor,
                                   instr_dropout_prob=args.distill_dropout_prob,
                                   reconstructor_dict=reconstructor_dict)
    if il_optimizer is not None:  # TODO: modify for same model
        for k, v in il_optimizer.items():
            il_trainer.optimizer_dict[k].load_state_dict(v.state_dict())

    sampler = MetaSampler(
        env=env,
        policy=policy_dict,
        rollouts_per_meta_task=args.rollouts_per_meta_task,
        meta_batch_size=10,
        max_path_length=args.max_path_length,
        parallel=not args.sequential,
        envs_per_task=1,
        obs_preprocessor=obs_preprocessor,
    )

    sample_processor = RL2SampleProcessor(
        discount=args.discount,
        gae_lambda=args.gae_lambda,
        normalize_adv=True,
        positive_adv=False,
    )

    envs = [copy.deepcopy(env) for _ in range(args.num_envs)]
    for i, new_env in enumerate(envs):
        new_env.update_distribution_from_other(env)
        new_env.seed(i)
        new_env.set_task()
        new_env.reset()
    augmenter = DataAugmenter(env.vocab()) if args.augment else None
    algo = PPOAlgo(policy_dict, envs, args, obs_preprocessor, augmenter, reconstructor_dict=reconstructor_dict)

    envs = [copy.deepcopy(env) for _ in range(args.num_envs)]
    algo_dagger = PPOAlgo(policy_dict, envs, args, obs_preprocessor, augmenter, reconstructor_dict=reconstructor_dict)

    if optimizer is not None:
        for k, v in optimizer.items():
            algo.optimizer_dict[k].load_state_dict(v.state_dict())

    EXP_NAME = get_exp_name(args)
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + "_" + str(args.seed)
    if original_saved_path is None and not args.continue_train:
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
    log_formats = ['stdout', 'log', 'csv']
    is_debug = args.prefix == 'DEBUG'

    if not is_debug:
        log_formats.append('tensorboard')
        # log_formats.append('wandb')
    logger.configure(dir=exp_dir, format_strs=log_formats,
                     snapshot_mode=args.save_option,
                     snapshot_gap=50, step=start_itr, name=args.prefix + str(args.seed), config=config)

    buffer_path = exp_dir if args.buffer_path is None else args.buffer_path

    # Log with the last teacher
    # Use the first teacher if we have one and aren't distilling to None
    # If we are distilling to none, keep that
    if args.self_distill and args.distillation_strategy in  ['all_teachers', 'no_teachers', 'powerset',
                                                             'single_teachers_none']:
        log_teacher = 'none'
    else:
        log_teacher = teachers_list[-2]  # Second to last (last is none)
    log_fn = make_log_fn(env, args, 0, exp_dir, log_teacher, True, seed=args.seed,
                         stochastic=True, num_rollouts=10, policy_name=EXP_NAME,
                         env_name=f'{args.env}-{teacher}-{args.level}',
                         log_every=10)


    trainer = Trainer(
        args,
        algo=algo,
        algo_dagger=algo_dagger,
        policy=policy_dict,
        env=deepcopy(env),
        sampler=sampler,
        sample_processor=sample_processor,
        start_itr=start_itr,
        buffer_name=buffer_path,
        exp_name=exp_dir,
        curriculum_step=curriculum_step,
        il_trainer=il_trainer,
        is_debug=is_debug,
        teacher_schedule=teacher_schedule,
        obs_preprocessor=obs_preprocessor,
        log_dict=log_dict,
        augmenter=augmenter,
        log_fn=log_fn,
    )
    trainer.train()


if __name__ == '__main__':
    base_path = 'data/'
    # if we want to sweep over any additional params, we can add them in
    sweep_params = {
    }
    DEFAULT = 'DEFAULT'
    parser = ArgumentParser()
    args = parser.parse_args()
    run_sweep(run_experiment, sweep_params, args.prefix, parser, 'c4.xlarge')
