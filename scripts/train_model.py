from algos.hierarchical_ppo_torch import HierarchicalPPOAgent
from algos.ppo import PPOAgent
from algos.sac import SACAgent
from algos.mf_trainer import Trainer
from scripts.arguments import ArgumentParser
from envs.babyai.utils.obs_preprocessor import make_obs_preprocessor
from scripts.test_generalization import make_log_fn
from algos.data_collector import DataCollector
from utils.rollout import rollout

import shutil
from logger import logger
from utils.utils import set_seed
from envs.babyai.levels.iclr19_levels import *
from envs.babyai.levels.envdist import EnvDist
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
        args.start_itr = 0
        log_dict = {}
    else:
        reload_path = args.reload_exp_path
        saved_model = joblib.load(reload_path + '/latest.pkl')
        args = saved_model['args']
        args.start_itr = saved_model['itr']
        args.buffer_path = args.exp_dir
        args.reload_exp_path = reload_path
        if args.rl_teacher is not None:
            args.rl_policy = reload_path
        if args.distill_teacher is not None:
            args.distill_policy = reload_path
        log_dict = saved_model.get('log_dict', {})
        set_seed(args.seed)
    return args, log_dict


def create_policy(path, teacher, env, args, obs_preprocessor):
    if args.algo == 'sac':
        args.on_policy = False
        agent = SACAgent(args=args, obs_preprocessor=obs_preprocessor, teacher=teacher, env=env, discount=args.discount,
                         init_temperature=args.entropy_coef, alpha_lr=args.lr, actor_lr=args.lr, critic_lr=args.lr,
                         control_penalty=args.control_penalty)
    elif args.algo == 'ppo':
        args.on_policy = True
        agent = PPOAgent(args=args, obs_preprocessor=obs_preprocessor, teacher=teacher, env=env)
    elif args.algo == 'hppo':
        args.on_policy = True
        agent = HierarchicalPPOAgent(args=args, obs_preprocessor=obs_preprocessor, teacher=teacher, env=env, discount=args.discount,
                                     lr=args.lr, control_penalty=args.control_penalty)
    else:
        raise NotImplementedError(args.algo)
    if path is not None:
        agent.load(path)
    return agent


def get_feedback_list(args):
    feedback_list = []
    if args.collect_teacher is not None:
        feedback_list.append(args.collect_teacher)
    if args.rl_teacher is not None:
        feedback_list.append(args.rl_teacher)
    if args.distill_teacher is not None:
        feedback_list.append(args.distill_teacher)
    if args.relabel_teacher is not None:
        feedback_list.append(args.relabel_teacher)
    return feedback_list

def make_env(args, feedback_list):
    if args.env in ['point_mass', 'ant', 'dummy']:
        args.no_instr = True
        args.discrete = False
        args.image_obs = False
        if args.discount == 'default':
            args.discount = .99
        else:
            args.discount = float(args.discount)
        if args.horizon in ['default', None]:
            args.horizon = None
        else:
            args.horizon = int(args.horizon)
    if args.env in ['babyai']:
        args.discrete = True
        args.image_obs = True
        args.no_instr = False
        args.fully_observed = True
        args.padding = True
        args.feedback_freq = [20]
        if args.discount == 'default':
            args.discount = .25
        else:
            args.discount = float(args.discount)
        if args.horizon == 'default':
            args.horizon = 200
        else:
            args.horizon = int(args.horizon)
        if args.train_level:
            args.env_dist = 'five_levels'
            args.leave_out_object = True
            args.level = 25
        if args.reward_type == 'default_reward':
            args.reward_type = 'dense'
    if args.env == 'point_mass':
        if args.train_level:
            args.level = 4
        if args.reward_type == 'default_reward':
            args.reward_type = 'waypoint'
    if args.env == 'ant':
        if args.train_level:
            args.level = 6
        if args.reward_type == 'default_reward':
            args.reward_type = 'vector_dir_waypoint'
    if args.env in ['dummy_discrete']:
        args.discrete = True
        args.image_obs = False
        args.no_instr = True

    arguments = {
        "start_loc": 'all',
        "include_holdout_obj": not args.leave_out_object,
        "feedback_type": feedback_list,
        "feedback_freq": args.feedback_freq,
        "reward_type": args.reward_type,
        "fully_observed": args.fully_observed,
        "padding": args.padding,
        "args": args,
        "seed": args.seed,
        "static_env": args.static_env,
        "horizon": args.horizon,
    }

    env = EnvDist(args.env_dist, env=args.env, start_index=args.level, **arguments)
    return env


def configure_logger(args, exp_dir, start_itr, is_debug):
    if args.reload_exp_path is None:
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
    log_formats = ['stdout', 'log', 'csv']

    if not (args.no_tb or is_debug):
        log_formats.append('tensorboard')
    logger.configure(dir=exp_dir, format_strs=log_formats,
                     snapshot_mode=args.save_option,
                     snapshot_gap=50, step=start_itr, name=args.prefix + str(args.seed))


def eval_policy(policy, env, args, exp_dir):
    save_dir = pathlib.Path(exp_dir)
    with open(save_dir.joinpath('results.csv'), 'w') as f:
        f.write('policy_env,policy,env,success_rate,stoch_accuracy,det_accuracy,reward\n')
    if not save_dir.exists():
        save_dir.mkdir()
    for env_index in args.eval_envs:
        print(f"Rolling out with env {env_index}")
        env.set_level_distribution(env_index)
        env.seed(args.seed)
        env.reset()
        policy.train(False)
        video_name = f'vids_env_{env_index}'
        paths, accuracy, stoch_accuracy, det_accuracy, reward = rollout(env, policy,
                                                                        instr_dropout_prob=int(args.hide_instrs)
                                                                        stochastic=True,
                                                                        record_teacher=True,
                                                                        video_directory=save_dir,
                                                                        video_name=video_name,
                                                                        num_rollouts=args.num_rollouts,
                                                                        save_wandb=False,
                                                                        save_locally=True,
                                                                        num_save=args.num_rollouts,
                                                                        rollout_oracle=False,
                                                                        teacher_name=policy.teacher)
        success_rate = np.mean([path['env_infos'][-1]['success'] for path in paths])
        try:
            success_rate = np.mean([path['env_infos'][-1]['timestep_success'] for path in paths])
        except:
            pass

        print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy},"
              f" reward: {reward}")
        with open(save_dir.joinpath('results.csv'), 'a') as f:
            f.write(f'{args.prefix}-{env_index},{args.prefix},{env_index},{success_rate},{stoch_accuracy},'
                    f'{det_accuracy},{reward} \n')


def run_experiment(args):
    args, log_dict = load_experiment(args)
    exp_name = args.prefix
    set_seed(args.seed)
    feedback_list = get_feedback_list(args)
    env = make_env(args, feedback_list)
    args.feedback_list = feedback_list
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
    if args.relabel_teacher is not None:
        relabel_policy = create_policy(args.relabel_policy, args.relabel_teacher, env, args, obs_preprocessor)
    else:
        relabel_policy = None

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

    exp_dir = os.getcwd() + '/logs/' + exp_name
    args.exp_dir = exp_dir
    is_debug = args.prefix == 'DEBUG'
    configure_logger(args, exp_dir, args.start_itr, is_debug)

    if args.eval_envs is not None:
        eval_policy(log_policy, env, args, exp_dir)
        return

    envs = [env.copy() for _ in range(args.num_envs)]
    for i, new_env in enumerate(envs):
        new_env.seed(i+100)
        new_env.set_task()
        new_env.reset()
    if collect_policy is None:
        sampler = None
    else:
        sampler = DataCollector(collect_policy, envs, args)

    buffer_name = exp_dir if args.buffer_path is None else args.buffer_path
    args.buffer_name = buffer_name
    num_rollouts = 1 if is_debug else args.num_rollouts
    log_fn = make_log_fn(env, args, 0, exp_dir, log_policy, hide_instrs=args.hide_instrs, seed=args.seed+1000,
                         stochastic=True, num_rollouts=num_rollouts, policy_name=exp_name,
                         env_name=str(args.level),
                         log_every=args.log_interval)

    trainer = Trainer(
        args=args,
        collect_policy=collect_policy,
        rl_policy=rl_policy,
        il_policy=il_policy,
        relabel_policy=relabel_policy,
        sampler=sampler,
        env=deepcopy(env),
        obs_preprocessor=obs_preprocessor,
        log_dict=log_dict,
        log_fn=log_fn,
    )
    trainer.train()


if __name__ == '__main__':
    from datetime import datetime
    import os
    try:
        parser = ArgumentParser()
        args = parser.parse_args()
        run_experiment(args)
        file = pathlib.Path('/home/olivia/failed_runs.txt')
        if file.exists():
            with open(file, 'a') as f:
                f.writelines(f'Finished run {args.prefix} on GPU {os.environ["CUDA_VISIBLE_DEVICES"]} at time {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
                f.writelines('\n' + "=" * 100 + '\n')
    except Exception as e:
        import traceback

        error_content = [
            f'Run Name: {args.prefix}',
            f'Time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}',
            f'GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}',
            f'Error: {traceback.format_exc()}',
            '=======================================================================================================\n',
        ]

        for error_line in error_content[:-2]:
            print(error_line)

        file = pathlib.Path('/home/olivia/failed_runs.txt')
        if file.exists():
            with open(file, 'a') as f:
                f.writelines(error_content)
        else:
            print("Not logging anywhere b/c we can't find the file")
        raise


