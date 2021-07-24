from algos.sac_torch import SACAgent
from algos.mf_trainer import Trainer
from scripts.arguments import ArgumentParser
from envs.babyai.utils.obs_preprocessor import make_obs_preprocessor
from scripts.test_generalization import make_log_fn
from algos.data_collector import DataCollector

import shutil
from logger import logger
from utils.utils import set_seed
from envs.babyai.levels.iclr19_levels import *
from envs.babyai.levels.curriculum import Curriculum
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


def load_experiment(args):
    if args.reload_exp_path is None:
        start_itr = 0
        curriculum_step = args.level
        log_dict = {}
    else:
        saved_model = joblib.load(args.reload_exp_path)
        args = saved_model['args']
        start_itr = saved_model['itr']
        curriculum_step = saved_model['curriculum_step']
        log_dict = saved_model.get('log_dict', {})
        set_seed(args.seed)
    return start_itr, curriculum_step, args, log_dict


def create_policy(path, teacher, env, args, obs_preprocessor):
    agent = SACAgent(args=args, obs_preprocessor=obs_preprocessor, teacher=teacher, env=env,
                    init_temperature=args.entropy_coef, alpha_lr=args.lr, actor_lr=args.lr, critic_lr=args.lr,
                     control_penalty=args.control_penalty)
    if path is not None:
        agent.load(path)
    return agent


def zero_thresholds(args):
    args.success_threshold_rl = 0
    args.success_threshold_rollout_teacher = 0
    args.success_threshold_rollout_no_teacher = 0
    args.accuracy_threshold_rl = 0
    args.accuracy_threshold_distill_teacher = 0
    args.accuracy_threshold_distill_no_teacher = 0
    args.accuracy_threshold_rollout_teacher = 0
    args.accuracy_threshold_rollout_no_teacher = 0

def get_feedback_list(args):
    feedback_list = []
    if args.collect_teacher is not None:
        feedback_list.append(args.collect_teacher)
    if args.rl_teacher is not None:
        feedback_list.append(args.rl_teacher)
    if args.distill_teacher is not None:
        feedback_list.append(args.distill_teacher)
    return feedback_list

def make_env(args, feedback_list):
    if args.env in ['point_mass', 'ant', 'dummy']:
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
        "feedback_type": feedback_list,
        "feedback_freq": args.feedback_freq,
        "cartesian_steps": args.cartesian_steps,
        "num_meta_tasks": args.rollouts_per_meta_task,
        "intermediate_reward": args.reward_type == 'dense',
        "reward_type": args.reward_type,
        "fully_observed": args.fully_observed,
        "padding": args.padding,
        "args": args,
        "seed": args.seed,
        "static_env": args.static_env,
    }
    if args.zero_all_thresholds:
        zero_thresholds(args)

    env = Curriculum(args.advance_curriculum_func, env=args.env, start_index=args.level,
                               curriculum_type=args.curriculum_type, **arguments)
    return env


def configure_logger(args, exp_dir, start_itr, is_debug):
    if not args.continue_train:
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
    log_formats = ['stdout', 'log', 'csv']

    if not is_debug:
        log_formats.append('tensorboard')
    logger.configure(dir=exp_dir, format_strs=log_formats,
                     snapshot_mode=args.save_option,
                     snapshot_gap=50, step=start_itr, name=args.prefix + str(args.seed))


def run_experiment():
    parser = ArgumentParser()
    args = parser.parse_args()
    exp_name = args.prefix

    set_seed(args.seed)
    start_itr, curriculum_step, args, log_dict = load_experiment(args)
    feedback_list = get_feedback_list(args)
    env = make_env(args, feedback_list)
    obs_preprocessor = make_obs_preprocessor(feedback_list)

    # Either we need an existing dataset, or we need to collect
    assert (args.buffer_path or (args.collect_policy is not None) or
            (args.rl_teacher is not None and args.collect_with_rl_policy) or
            (args.distill_teacher is not None and args.collect_with_distill_policy))
    # We can't collect with both policies
    assert not (args.collect_with_rl_policy and args.collect_with_distill_policy)


    log_policy = None
    if args.rl_teacher is not None:
        rl_policy = create_policy(args.rl_policy, args.rl_teacher, env, args,
                                  obs_preprocessor)
        log_policy = rl_policy
    else:
        rl_policy = None
    if args.distill_teacher is not None:
        il_policy = create_policy(args.distill_policy, args.distill_teacher, env, args, obs_preprocessor)
        log_policy = il_policy
    else:
        il_policy = None

    if args.collect_with_rl_policy:
        collect_policy = rl_policy
        args.collect_teacher = args.rl_teacher
    elif args.collect_with_distill_policy:
        collect_policy = il_policy
        args.collect_teacher = args.distill_teacher
    elif args.collect_teacher is not None:
        collect_policy = create_policy(args.collect_policy, args.collect_teacher, env, args, obs_preprocessor)
        if log_policy is None:
            log_policy = collect_policy
    else:
        collect_policy = None

    envs = [env.copy() for _ in range(args.num_envs)]
    for i, new_env in enumerate(envs):
        new_env.seed(i)
        new_env.set_task()
        new_env.reset()
    sampler = DataCollector(collect_policy, envs, args, obs_preprocessor)

    exp_dir = os.getcwd() + '/logs/data/' + exp_name + "_" + str(args.seed)
    args.exp_dir = exp_dir
    is_debug = args.prefix == 'DEBUG'
    configure_logger(args, exp_dir, start_itr, is_debug)

    buffer_path = exp_dir if args.buffer_path is None else args.buffer_path
    num_rollouts = 1 #if is_debug else 10
    log_fn = make_log_fn(env, args, 0, exp_dir, log_policy, True, seed=args.seed,
                         stochastic=True, num_rollouts=num_rollouts, policy_name=exp_name,
                         env_name=str(args.level),
                         log_every=10)

    trainer = Trainer(
        args=args,
        collect_policy=collect_policy,
        rl_policy=rl_policy,
        il_policy=il_policy,
        sampler=sampler,
        env=deepcopy(env),
        start_itr=start_itr,
        buffer_name=buffer_path,
        curriculum_step=curriculum_step,
        obs_preprocessor=obs_preprocessor,
        log_dict=log_dict,
        log_fn=log_fn,
        feedback_list=feedback_list,
    )
    trainer.train()


if __name__ == '__main__':
    base_path = 'data/'
    run_experiment()

