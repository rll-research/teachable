# imports
import joblib
import os
import copy
import numpy as np
import argparse
import pathlib
import time

from meta_mb.samplers.utils import rollout
from meta_mb.logger import logger
from babyai.utils.obs_preprocessor import make_obs_preprocessor
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_policy(path):
    saved_model = joblib.load(path)
    env = saved_model['env']
    policy = saved_model['policy']
    args = saved_model['args']
    if type(policy) is dict:
        for p_dict in policy.values():
            if hasattr(p_dict, 'instr_rnn'):  # Runs faster (for versions of the model with rnns)
                p_dict.instr_rnn.flatten_parameters()
    else:
        raise NotImplementedError("Change the code back to not using separate dicts. Change was made on 1/31/21")
        if 'supervised_model' in saved_model:
            supervised_model = copy.deepcopy(saved_model['supervised_model'])  # Deepcopy to check it's not the policy
        else:
            supervised_model = copy.deepcopy(policy)
        try:
            policy.instr_rnn.flatten_parameters()
            supervised_model.instr_rnn.flatten_parameters()
        except Exception as e:
            print(e, "looks like instrs aren't rnn")
        assert not policy is supervised_model
        is_dict = False
    return policy, env, args, saved_model


def eval_policy(env, policy, save_dir, num_rollouts, teachers, hide_instrs, stochastic, args, seed=0,
                video_name='generalization_vids', num_save=20):
    if not save_dir.exists():
        save_dir.mkdir()
    env.seed(seed)
    env.reset()
    if teachers == ['all']:
        teacher_dict = {f: True for f in env.feedback_type}
    elif teachers == ['none']:
        teacher_dict = {f: False for f in env.feedback_type}
    else:
        teacher_dict = {f: f in teachers for f in env.feedback_type}
    try:
        teacher_null_dict = env.teacher.null_feedback()
    except Exception as e:
        teacher_null_dict = {}
    obs_preprocessor = make_obs_preprocessor(teacher_null_dict, include_zeros=args.include_zeros)
    policy.eval()
    paths, accuracy, stoch_accuracy, det_accuracy, reward = rollout(env, policy,
                                                                    instrs=not hide_instrs,
                                                                    reset_every=1,
                                                                    stochastic=stochastic,
                                                                    record_teacher=True,
                                                                    teacher_dict=teacher_dict,
                                                                    teacher_name=teachers[0],
                                                                    video_directory=save_dir,
                                                                    video_name=video_name,
                                                                    num_rollouts=num_rollouts,
                                                                    save_wandb=False,
                                                                    save_locally=num_save > 0,
                                                                    num_save=num_save,
                                                                    obs_preprocessor=obs_preprocessor,
                                                                    rollout_oracle=False)
    success_rate = np.mean([path['env_infos'][-1]['success'] for path in paths])
    try:
        success_rate = np.mean([path['env_infos'][-1]['timestep_success'] for path in paths])
    except:
        print("doesn't have timestep_success")
    try:
        teacher_actions = [np.array([timestep['teacher_action'][0] for timestep in path['env_infos']]) for path in
                           paths]
        agent_actions = [np.array(path['actions']) for path in paths]
        errors = [np.sum(1 - (teacher_a == agent_a)) / len(teacher_a) for teacher_a, agent_a in
                  zip(teacher_actions, agent_actions)]
        plt.hist(errors)
        plt.title(f"Distribution of errors {str(teachers)}")
        plt.savefig(save_dir.joinpath('errors.png'))
    except:
        print("No teacher, so can't plot errors")
    return success_rate, stoch_accuracy, det_accuracy, reward


def make_log_fn(env, args, start_num_feedback, save_dir, teacher, hide_instrs, seed=1, stochastic=True,
                num_rollouts=10, policy_name='policy', env_name='env', log_every=10):
    start = time.time()
    save_dir = pathlib.Path(save_dir)

    def log_fn_vidrollout(policy, itr, num_save):
        return test_success_checkpoint(env, save_dir, num_rollouts, [teacher], policy=policy, policy_name=policy_name,
                                       env_name=env_name, hide_instrs=hide_instrs, itr=itr, stochastic=stochastic,
                                       args=args,
                                       seed=seed, num_save=num_save)

    def log_fn(policy, logger, itr, num_feedback):
        policy_env_name = f'Policy{policy_name}-{env_name}'
        full_save_dir = save_dir.joinpath(policy_env_name + f'_checkpoint{seed}')
        if itr == 0:
            if not full_save_dir.exists():
                full_save_dir.mkdir()
            with open(full_save_dir.joinpath('results.csv'), 'w') as f:
                f.write('policy_env,policy,env,success_rate,stoch_accuracy,itr,num_feedback,time,reward\n')
        if not itr % log_every == 0:
            return
        policy = policy[teacher]
        num_save = 0
        avg_success, avg_accuracy, det_accuracy, reward = log_fn_vidrollout(policy, itr, num_save)
        print(f"Finetuning achieved success: {avg_success}, stoch acc: {avg_accuracy}")
        with open(full_save_dir.joinpath('results.csv'), 'a') as f:
            f.write(
                f'{policy_env_name},{policy_name},{env_name},{avg_success},{avg_accuracy},{itr},'
                f'{num_feedback + start_num_feedback},{time.time() - start},{reward} \n')
        return avg_success, avg_accuracy

    return log_fn


def finetune_policy(env, env_index, policy, save_name, args, teacher_null_dict,
                    save_dir=pathlib.Path("."), teachers={}, policy_name="", env_name="",
                    hide_instrs=False, heldout_env=None, stochastic=True, num_rollouts=1, model_data={}, seed=0,
                    start_num_feedback=0):
    # Normally we would put the imports up top, but we also import this file in Trainer
    # Importing here prevents us from getting stuck in infinite loops
    from meta_mb.algos.ppo_torch import PPOAlgo
    from meta_mb.trainers.mf_trainer import Trainer
    from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
    from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
    from meta_mb.trainers.il_trainer import ImitationLearning
    from babyai.teacher_schedule import make_teacher_schedule

    # TODO: consider deleting this!
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
        # "intermediate_reward": not args.sparse_reward,
    }
    # curriculum_step = 26  # TODO: don't hardcode this!
    # env = rl2env(normalize(Curriculum(args.advance_curriculum_func, start_index=curriculum_step,
    #                                   curriculum_type=args.curriculum_type, **arguments)
    #                        ), ceil_reward=args.ceil_reward)

    obs_preprocessor = make_obs_preprocessor(teacher_null_dict, include_zeros=args.include_zeros)

    if args.repeated_seed:
        print("using repeated seed")
        args.num_envs = num_rollouts
    args.model = 'default_il'
    if hasattr(args, 'instr_dropout_prob'):
        # TODO: remove this once the current runs are irrelevant
        args.distill_dropout_prob = args.instr_dropout_prob
    il_trainer = ImitationLearning(policy, env, args, distill_with_teacher=False,
                                   preprocess_obs=obs_preprocessor, label_weightings=args.distill_label_weightings,
                                   instr_dropout_prob=args.distill_dropout_prob)
    try:
        if 'il_optimizer' in model_data:
            for k, optimizer in model_data['il_optimizer'].items():
                il_trainer.optimizer_dict[k].load_state_dict(optimizer.state_dict())
    except Except as e:
        print("couldn't load il optimizer", e)
    rp_trainer = None
    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=args.rollouts_per_meta_task,
        meta_batch_size=10,
        max_path_length=args.max_path_length,
        parallel=not args.sequential,
        envs_per_task=1,
        reward_predictor=None,
        obs_preprocessor=obs_preprocessor,
    )

    sample_processor = RL2SampleProcessor(
        discount=args.discount,
        gae_lambda=args.gae_lambda,
        normalize_adv=True,
        positive_adv=False,
    )
    envs = [copy.deepcopy(env) for _ in range(args.num_envs)]
    offset = seed
    for i, new_env in enumerate(envs):
        new_env.update_distribution_from_other(env)
        new_env.seed(i + offset * 100)
        new_env.set_task()
        new_env.reset()
    repeated_seed = None if not args.repeated_seed else np.arange(1000 * seed, 1000 * seed + args.num_envs)
    collect_dropout_prob = 1 if hide_instrs else 0
    if args.env in ['point_mass', 'ant']:
        args.no_instr = True
        discrete = False
    elif args.env in ['babyai']:
        discrete = True
    else:
        raise NotImplementedError(f'Unknown env {args.env}')
    args.discrete = discrete
    algo = PPOAlgo(policy, envs, args, obs_preprocessor, None)

    try:
        if 'optimizer' in model_data:
            for k, optimizer in model_data['optimizer'].items():
                algo.optimizer_dict[k].load_state_dict(optimizer.state_dict())

    except Exception as e:
        print("Couldn't load normal optimizer", e)
    teacher_schedule = make_teacher_schedule(args.feedback_type, args.teacher_schedule)
    # Standardize args
    args.single_level = True
    args.reward_when_necessary = False  # TODO: make this a flag

    log_fn = make_log_fn(env, args, start_num_feedback, save_dir, teachers[0], hide_instrs, seed=seed,
                         stochastic=stochastic, num_rollouts=num_rollouts, policy_name=policy_name, env_name=env_name,
                         log_every=args.log_every)

    log_formats = ['stdout', 'log', 'csv', 'tensorboard']
    logger.configure(dir=save_name, format_strs=log_formats,
                     snapshot_mode=args.save_option,
                     snapshot_gap=50, step=0, name=args.prefix + str(args.seed), config={})
    trainer = Trainer(
        args,
        algo=algo,
        algo_dagger=algo,
        policy=policy,
        env=copy.deepcopy(env),
        sampler=sampler,
        sample_processor=sample_processor,
        buffer_name=args.buffer_name if args.buffer_name is not None else save_name,
        exp_name=save_name,
        curriculum_step=env_index,
        il_trainer=il_trainer,
        reward_predictor=None,
        rp_trainer=rp_trainer,
        is_debug=False,
        teacher_schedule=teacher_schedule,
        obs_preprocessor=obs_preprocessor,
        log_dict={},
        log_and_save=True,  # False,
        eval_heldout=False,
        log_fn=log_fn,
        log_every=1,
    )
    print("TRAINING!!!")
    trainer.train()
    print("TOTAL FEEDBACK", trainer.num_feedback_reward / 800, trainer.num_feedback_advice / 800)
    print("All done!")
    return trainer


def test_success(env, env_index, save_dir, num_rollouts, teachers, teacher_null_dict, policy_path=None, policy=None,
                 policy_name="", env_name="", hide_instrs=False, heldout_env=[], stochastic=True, additional_args={},
                 seed=0):
    if policy is None:
        policy, _, args, model_data = load_policy(policy_path)
        for k, v in additional_args.items():
            if not hasattr(args, k) or (not v is None):
                setattr(args, k, v)
        if additional_args['target_policy'] is not None:
            policy[args.target_policy_key] = load_policy(args.target_policy)[0][args.target_policy_key]
        n_itr = args.n_itr
    else:
        n_itr = 0
    policy_env_name = f'Policy{policy_name}-{env_name}'
    print("EVALUATING", policy_env_name)
    full_save_dir = save_dir.joinpath(policy_env_name)
    if not full_save_dir.exists():
        full_save_dir.mkdir()
    if n_itr > 0:
        finetune_path = full_save_dir.joinpath(f'finetuned_policy{seed}')
        if not finetune_path.exists():
            finetune_path.mkdir()
        args.seed = seed
        num_feedback = 0
        if args.finetune_teacher_first > 0:
            finetune_teacher_args = copy.deepcopy(args)
            finetune_teacher_args.n_itr = args.finetune_teacher_first
            finetune_teacher_args.teacher_schedule = 'first_teacher'
            finetune_teacher_args.distillation_strategy = 'single_teachers'
            finetune_teacher_args.yes_distill = True
            finetune_teacher_args.no_distill = False
            if seed is not None:
                finetune_teacher_path = full_save_dir.joinpath(f'finetuned_teachers{seed}')
                if not finetune_teacher_path.exists():
                    finetune_teacher_path.mkdir()
                    if not finetune_teacher_path.exists():
                        finetune_teacher_path.mkdir()
                    with open(finetune_teacher_path.joinpath('results.csv'), 'w') as f:
                        f.write('policy_env,policy,env,success_rate,stoch_accuracy,itr,num_feedback\n')
                finetune_teacher_args.seed = seed
            trainer = finetune_policy(env, env_index, policy,
                                      finetune_teacher_path, finetune_teacher_args, teacher_null_dict,
                                      save_dir=save_dir, teachers=teachers, policy_name=policy_name, env_name=env_name,
                                      hide_instrs=hide_instrs, heldout_env=heldout_env, stochastic=stochastic,
                                      num_rollouts=num_rollouts, model_data=model_data, seed=seed)
            num_feedback = trainer.num_feedback_advice + trainer.num_feedback_reward
            # policy, _, _, _ = load_policy(finetune_path.joinpath('latest.pkl'))  # TODO: this might be important!
            if args.target_policy is not None:
                policy[args.target_policy_key] = load_policy(args.target_policy)[0][args.target_policy_key]
        finetune_policy(env, env_index, policy,
                        finetune_path, args, teacher_null_dict,
                        save_dir=save_dir, teachers=teachers, policy_name=policy_name, env_name=env_name,
                        hide_instrs=hide_instrs, heldout_env=heldout_env, stochastic=stochastic,
                        num_rollouts=num_rollouts, model_data=model_data, seed=seed, start_num_feedback=num_feedback)
    assert len(teachers) == 1
    teacher_policy = policy[teachers[0]]
    success_rate, stoch_accuracy, det_accuracy, followed_cc3 = eval_policy(env, teacher_policy, full_save_dir,
                                                                           num_rollouts,
                                                                           teachers, hide_instrs, stochastic, args,
                                                                           seed)
    print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    with open(save_dir.joinpath('results.csv'), 'a') as f:
        f.write(
            f'{policy_env_name},{policy_name},{env_name},{success_rate},{stoch_accuracy},{det_accuracy},{followed_cc3} \n')
    return success_rate, stoch_accuracy, det_accuracy


def test_success_checkpoint(env, save_dir, num_rollouts, teachers, policy=None,
                            policy_name="", env_name="", hide_instrs=False, itr=-1, stochastic=True, args=None,
                            seed=0, num_save=0):
    policy_env_name = f'Policy{policy_name}-{env_name}'
    full_save_dir = save_dir.joinpath(policy_env_name + f'_checkpoint{seed}')
    if not full_save_dir.exists():
        full_save_dir.mkdir()
    success_rate, stoch_accuracy, det_accuracy, reward = eval_policy(env, policy, full_save_dir, num_rollouts,
                                                                     teachers, hide_instrs, stochastic, args,
                                                                     seed, f'vid_{itr}', num_save=num_save)
    print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    return success_rate, stoch_accuracy, det_accuracy, reward


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument('--target_policy', type=str, default=None)
    parser.add_argument('--target_policy_key', type=str, default='none')
    parser.add_argument('--envs', nargs='+', required=True, type=str)
    parser.add_argument('--levels', nargs='+', default=['latest'], type=str)
    parser.add_argument('--teachers', nargs='+', default=['all'], type=str)
    parser.add_argument("--finetune_itrs", default=0, type=int)
    parser.add_argument("--min_itr_steps_distill", default=0, type=int)
    parser.add_argument("--num_rollouts", default=50, type=int)
    parser.add_argument("--no_train_rl", action='store_true')
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--hide_instrs", action='store_true')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument('--teacher_schedule', type=str, default='last_teacher')
    parser.add_argument('--distillation_strategy', type=str, choices=[
        'all_teachers', 'no_teachers', 'all_but_none', 'powerset', 'single_teachers', 'single_teachers_none'
    ], default='single_teachers_none')
    parser.add_argument('--no_distill', action='store_true')
    parser.add_argument('--yes_distill', action='store_true')
    parser.add_argument('--rollout_temperature', type=float, default=1)
    parser.add_argument('--finetune_il', action='store_true')
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--finetune_teacher_first', type=int, default=0)
    parser.add_argument('--repeated_seed', action='store_true')
    parser.add_argument('--distillation_steps', type=int, default=None)
    parser.add_argument('--seeds', nargs='+', default=[0], type=int)
    parser.add_argument('--distill_successful_only', action='store_true')
    parser.add_argument('--buffer_name', default=None)
    parser.add_argument('--collect_with_oracle', action='store_true')
    parser.add_argument('--frames_per_proc', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    save_dir = pathlib.Path(args.save_dir)
    policy_path = pathlib.Path(args.policy)

    _, default_env, config, model_data = load_policy(policy_path.joinpath(args.levels[0] + '.pkl'))
    default_env.reset()
    try:
        teacher_null_dict = default_env.teacher.null_feedback()
    except:
        teacher_null_dict = {}

    # Get the levels of the policies to load
    policy_levels = args.levels
    if policy_levels == ['all']:
        policy_levels = range(len(default_env.levels_list))
    policy_level_names = []
    for policy_level in policy_levels:
        try:
            level_number = int(policy_level)
            policy_level_names.append(f'level_{level_number}.pkl')
        except ValueError:
            if not policy_level[-4:] == '.pkl':
                policy_level = policy_level + '.pkl'
            policy_level_names.append(policy_level)

    # Get the levels of the envs to test on
    env_names = args.envs
    env_indices = []
    num_train_envs = len(default_env.train_levels)
    num_test_envs = len(default_env.held_out_levels)
    for env_name in env_names:
        if env_name == 'train':
            env_indices += list(range(num_train_envs))
        elif env_name == 'test':
            env_indices += list(range(num_train_envs, num_train_envs + num_test_envs))
        elif 'test' == env_name[:4]:
            index = int(env_name[4:])
            # Test levels start directly after train levels, so add the length of the train levels list
            env_indices.append(index + num_train_envs)
        else:
            try:
                env_id = int(env_name)
                env_indices.append(env_id)
            except ValueError:
                for i, level in enumerate(default_env.levels_list):
                    if env_name in level.__class__.__name__:
                        env_indices.append(i)
    envs = []
    for env_index in env_indices:
        env = copy.deepcopy(default_env)
        env.set_level_distribution(env_index)
        env.set_task()
        env.reset()
        envs.append((env, env_index))

    additional_args = {}
    additional_args['n_itr'] = args.finetune_itrs
    additional_args['teacher_schedule'] = args.teacher_schedule
    additional_args['distillation_strategy'] = args.distillation_strategy
    additional_args['no_train_rl'] = args.no_train_rl
    additional_args['no_rollouts'] = True
    additional_args['yes_rollouts'] = False
    additional_args['no_collect'] = False
    additional_args['yes_distill'] = args.yes_distill
    additional_args['no_distill'] = args.no_distill
    additional_args['rollout_temperature'] = args.rollout_temperature
    additional_args['finetune_il'] = args.finetune_il
    additional_args['log_every'] = args.log_every
    additional_args['finetune_teacher_first'] = args.finetune_teacher_first
    additional_args['repeated_seed'] = args.repeated_seed
    if args.distillation_steps is not None:
        additional_args['distillation_steps'] = args.distillation_steps
    additional_args['target_policy'] = args.target_policy
    additional_args['target_policy_key'] = args.target_policy_key
    additional_args['distill_successful_only'] = args.distill_successful_only
    additional_args['buffer_name'] = args.buffer_name
    additional_args['collect_with_oracle'] = args.collect_with_oracle
    additional_args['source'] = 'agent'  # TODO: remove
    additional_args['frames_per_proc'] = args.frames_per_proc
    additional_args['batch_size'] = args.batch_size
    additional_args['lr'] = args.lr
    additional_args['min_itr_steps_distill'] = args.min_itr_steps_distill
    if args.collect_with_oracle:
        additional_args['source'] = 'teacher'
    if args.buffer_name is not None:
        additional_args['no_collect'] = True
        additional_args['source'] = 'agent'
        additional_args['feedback_from_buffer'] = True

    # Test every policy with every level
    if not save_dir.exists():
        save_dir.mkdir()
    with open(save_dir.joinpath('results.csv'), 'w') as f:
        f.write('policy_env,policy, env,success_rate, stoch_accuracy, det_accuracy, followed_cc3 \n')
    for policy_name in policy_level_names:
        for env, env_index in envs:
            inner_env = env
            while hasattr(inner_env, '_wrapped_env'):
                inner_env = inner_env._wrapped_env
            for seed in args.seeds:
                test_success(env, env_index, save_dir, args.num_rollouts, args.teachers, teacher_null_dict,
                             policy_path=policy_path.joinpath(policy_name),
                             policy_name=policy_path.stem, env_name=str(env_index),  # inner_env.__class__.__name__,
                             hide_instrs=args.hide_instrs, heldout_env=env, stochastic=not args.deterministic,
                             additional_args=additional_args, seed=seed)


if __name__ == '__main__':
    main()
