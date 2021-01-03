from meta_mb.meta_envs.rl2_env import rl2env
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.ppo_torch import PPOAlgo
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
from babyai.model import ACModel
from meta_mb.trainers.il_trainer import ImitationLearning
from babyai.arguments import ArgumentParser
from babyai.utils.obs_preprocessor import make_obs_preprocessor
from babyai.teacher_schedule import make_teacher_schedule

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
    policy = saved_model['policy']
    optimizer = saved_model.get('optimizer', None)
    policy.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO: is this necessary?
    policy.hidden_state = None
    if original_config.continue_train is True:
        start_itr = saved_model['itr']
        curriculum_step = saved_model['curriculum_step']
    else:
        start_itr = 0
        curriculum_step = args.level

    reward_predictor = saved_model['reward_predictor']
    reward_predictor.hidden_state = None
    if 'supervised_model' in saved_model:
        # The supervised model can either be the same model as the policy or a different model
        if args.self_distill and args.distill_same_model:
            supervised_model = policy
        elif args.self_distill:
            supervised_model = saved_model['supervised_model']
        else:
            supervised_model = None
    else:
        supervised_model = None
    il_optimizer = saved_model.get('il_optimizer', None)
    log_dict = saved_model.get('log_dict', {})
    return policy, supervised_model, reward_predictor, optimizer, start_itr, curriculum_step, args, \
        il_optimizer, log_dict


def run_experiment(**config):
    parser = ArgumentParser()
    args = parser.parse_args()
    for k, v in config.items():
        setattr(args, k, v)

    set_seed(args.seed)
    original_saved_path = args.saved_path
    if original_saved_path is not None:
        policy, supervised_model, reward_predictor, optimizer, start_itr, curriculum_step, args, \
            il_optimizer, log_dict = load_model(args)
    else:
        il_optimizer = None
        log_dict = {}
    arguments = {
        "start_loc": 'all',
        "include_holdout_obj": True,
        "persist_goal": not args.reset_goal,
        "persist_objs": not args.reset_objs,
        "persist_agent": not args.reset_agent,
        "feedback_type": args.feedback_type,
        "feedback_freq": args.feedback_freq,
        "cartesian_steps": args.cartesian_steps,
        "num_meta_tasks": args.rollouts_per_meta_task,
        "intermediate_reward": args.intermediate_reward,
    }
    teacher_schedule = make_teacher_schedule(args.feedback_type, args.teacher_schedule)
    teacher_train_dict, _, _ = teacher_schedule(0, 0, 0)
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
        env = rl2env(normalize(Curriculum(args.advance_curriculum_func, start_index=curriculum_step,
                                          curriculum_type=args.curriculum_type, **arguments)
                               ), ceil_reward=args.ceil_reward)
    else:
        optimizer = None
        env = rl2env(normalize(Curriculum(args.advance_curriculum_func, start_index=args.level,
                                          curriculum_type=args.curriculum_type, **arguments)),
                     ceil_reward=args.ceil_reward)
        obs = env.reset()
        advice_size = sum([np.prod(obs[k].shape) for k in teacher_train_dict.keys()])
        if args.no_teacher:
            advice_size = 0
        policy = ACModel(action_space=env.action_space,
                         env=env,
                         image_dim=args.image_dim,
                         memory_dim=args.memory_dim,
                         instr_dim=args.instr_dim,
                         lang_model=args.instr_arch,
                         use_instr=not args.no_instr,
                         use_memory=not args.no_mem,
                         arch=args.arch,
                         advice_dim=args.advice_dim,
                         advice_size=advice_size,
                         num_modules=args.num_modules)

        reward_predictor = ACModel(action_space=spaces.Discrete(2),
                                   env=env,
                                   image_dim=args.image_dim,
                                   memory_dim=args.memory_dim,
                                   instr_dim=args.instr_dim,
                                   lang_model=args.instr_arch,
                                   use_instr=not args.no_instr,
                                   use_memory=not args.no_mem,
                                   arch=args.arch,
                                   advice_dim=args.advice_dim,
                                   advice_size=advice_size,
                                   num_modules=args.num_modules)
        if args.self_distill and not args.distill_same_model:
            supervised_model = ACModel(action_space=env.action_space,
                                       env=env,
                                       image_dim=args.image_dim,
                                       memory_dim=args.memory_dim,
                                       instr_dim=args.instr_dim,
                                       lang_model=args.instr_arch,
                                       use_instr=not args.no_instr,
                                       use_memory=not args.no_mem,
                                       arch=args.arch,
                                       advice_dim=args.advice_dim,
                                       advice_size=advice_size,
                                       num_modules=args.num_modules)
        elif args.self_distill:
            supervised_model = policy
        else:
            supervised_model = None
        start_itr = 0
        curriculum_step = env.index

    env.reset()
    try:
        teacher_null_dict = env.teacher.null_feedback()
    except Exception as e:
        teacher_null_dict = {}
    obs_preprocessor = make_obs_preprocessor(teacher_null_dict)
    if args.new_distill:
        print("ADDING SUP!!!")
        obs = env.reset()
        advice_size = sum([np.prod(obs[k].shape) for k in teacher_train_dict.keys()])
        supervised_model = ACModel(action_space=env.action_space,
                                   env=env,
                                   image_dim=args.image_dim,
                                   memory_dim=args.memory_dim,
                                   instr_dim=args.instr_dim,
                                   lang_model=args.instr_arch,
                                   use_instr=not args.no_instr,
                                   use_memory=not args.no_mem,
                                   arch=args.arch,
                                   advice_dim=args.advice_dim,
                                   advice_size=advice_size,
                                   num_modules=args.num_modules)

    args.model = 'default_il'
    if supervised_model is not None:
        modify_cc3_steps = args.cartesian_steps if args.modify_cc3 else None
        il_trainer = ImitationLearning(supervised_model, env, args, distill_with_teacher=False,
                                       preprocess_obs=obs_preprocessor, label_weightings=args.distill_label_weightings,
                                       instr_dropout_prob=args.instr_dropout_prob, modify_cc3_steps=modify_cc3_steps)
        if il_optimizer is not None:
            il_trainer.optimizer.load_state_dict(il_optimizer)
    else:
        il_trainer = None
    rp_trainer = ImitationLearning(reward_predictor, env, args, distill_with_teacher=True, reward_predictor=True,
                                   preprocess_obs=obs_preprocessor, label_weightings=args.distill_label_weightings)

    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=args.rollouts_per_meta_task,
        meta_batch_size=10,
        max_path_length=args.max_path_length,
        parallel=not args.sequential,
        envs_per_task=1,
        reward_predictor=reward_predictor,
        supervised_model=supervised_model,
        obs_preprocessor=obs_preprocessor,
    )

    sample_processor = RL2SampleProcessor(
        discount=args.discount,
        gae_lambda=args.gae_lambda,
        normalize_adv=True,
        positive_adv=False,
    )

    envs = [copy.deepcopy(env) for _ in range(args.num_envs)]
    algo = PPOAlgo(policy, envs, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                   args.gae_lambda,
                   args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                   args.optim_eps, args.clip_eps, args.epochs, args.meta_batch_size,
                   parallel=not args.sequential, rollouts_per_meta_task=args.rollouts_per_meta_task,
                   obs_preprocessor=obs_preprocessor)


    envs = [copy.deepcopy(env) for _ in range(args.num_envs)]
    algo_dagger = PPOAlgo(policy, envs, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                   args.gae_lambda,
                   args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                   args.optim_eps, args.clip_eps, args.epochs, args.meta_batch_size,
                   parallel=not args.sequential, rollouts_per_meta_task=args.rollouts_per_meta_task,
                   obs_preprocessor=obs_preprocessor)

    if optimizer is not None:
        algo.optimizer.load_state_dict(optimizer)

    EXP_NAME = get_exp_name(args)
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + "_" + str(args.seed)
    if original_saved_path is None and not args.continue_train:
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
    log_formats = ['stdout', 'log', 'csv']
    is_debug = args.prefix == 'DEBUG'

    if not is_debug:
        log_formats.append('tensorboard')
        log_formats.append('wandb')
    logger.configure(dir=exp_dir, format_strs=log_formats,
                     snapshot_mode=args.save_option,
                     snapshot_gap=50, step=start_itr, name=args.prefix + str(args.seed), config=config)

    buffer_path = exp_dir if args.buffer_path is None else args.buffer_path
    trainer = Trainer(
        args,
        algo=algo,
        algo_dagger=algo_dagger,
        policy=policy,
        env=deepcopy(env),
        sampler=sampler,
        sample_processor=sample_processor,
        start_itr=start_itr,
        buffer_name=buffer_path,
        exp_name=exp_dir,
        curriculum_step=curriculum_step,
        il_trainer=il_trainer,
        supervised_model=supervised_model,
        reward_predictor=reward_predictor,
        rp_trainer=rp_trainer,
        is_debug=is_debug,
        teacher_schedule=teacher_schedule,
        obs_preprocessor=obs_preprocessor,
        log_dict=log_dict,
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
