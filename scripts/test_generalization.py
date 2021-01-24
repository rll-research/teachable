# imports
import joblib
import os
import copy
import numpy as np
import argparse
import pathlib

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
    return policy, supervised_model, env, args, saved_model


def eval_policy(env, policy, save_dir, num_rollouts, teachers, hide_instrs, stochastic,
                video_name='generalization_vids'):
    if not save_dir.exists():
        save_dir.mkdir()
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
    obs_preprocessor = make_obs_preprocessor(teacher_null_dict)
    policy.eval()
    paths, accuracy, stoch_accuracy, det_accuracy, followed_cc3 = rollout(env, policy,
                                                                          instrs=not hide_instrs,
                                                                          reset_every=1,
                                                                          stochastic=stochastic,
                                                                          record_teacher=True,
                                                                          teacher_dict=teacher_dict,
                                                                          video_directory=save_dir,
                                                                          video_name=video_name,
                                                                          num_rollouts=num_rollouts,
                                                                          save_wandb=False,
                                                                          save_locally=True,
                                                                          num_save=num_rollouts,
                                                                          obs_preprocessor=obs_preprocessor,
                                                                          rollout_oracle=False)
    success_rate = np.mean([path['env_infos'][-1]['success'] for path in paths])
    teacher_actions = [np.array([timestep['teacher_action'][0] for timestep in path['env_infos']]) for path in paths]
    agent_actions = [np.array(path['actions']) for path in paths]
    errors = [np.sum(1 - (teacher_a == agent_a))/len(teacher_a) for teacher_a, agent_a in zip(teacher_actions, agent_actions)]
    plt.hist(errors)
    plt.title(f"Distribution of errors {str(teachers)}")
    plt.savefig(save_dir.joinpath('errors.png'))
    return success_rate, stoch_accuracy, det_accuracy, followed_cc3


def finetune_policy(env, env_index, policy, supervised_model, save_name, args, teacher_null_dict,
                    save_dir=pathlib.Path("."), teachers={}, policy_name="", env_name="",
                    hide_instrs=False, heldout_env=None, stochastic=True, num_rollouts=1, model_data={}):
    # Normally we would put the imports up top, but we also import this file in Trainer
    # Importing here prevents us from getting stuck in infinite loops
    from meta_mb.algos.ppo_torch import PPOAlgo
    from meta_mb.trainers.mf_trainer import Trainer
    from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
    from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
    from meta_mb.trainers.il_trainer import ImitationLearning
    from babyai.teacher_schedule import make_teacher_schedule
    # from meta_mb.meta_envs.rl2_env import rl2env
    # from meta_mb.envs.normalized_env import normalize
    # from babyai.levels.curriculum import Curriculum

    if args.finetune_il:
        policy = copy.deepcopy(supervised_model)

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
        "intermediate_reward": not args.sparse_reward,
    }
    # curriculum_step = 26  # TODO: don't hardcode this!
    # env = rl2env(normalize(Curriculum(args.advance_curriculum_func, start_index=curriculum_step,
    #                                   curriculum_type=args.curriculum_type, **arguments)
    #                        ), ceil_reward=args.ceil_reward)

    obs_preprocessor = make_obs_preprocessor(teacher_null_dict)

    args.model = 'default_il'
    if supervised_model is not None:
        il_trainer = ImitationLearning(supervised_model, env, args, distill_with_teacher=False,
                                       preprocess_obs=obs_preprocessor, label_weightings=args.distill_label_weightings,
                                       instr_dropout_prob=args.instr_dropout_prob)
        if 'il_optimizer' in model_data:
            il_trainer.optimizer.load_state_dict(model_data['il_optimizer'])
    else:
        il_trainer = None
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
    for i, new_env in enumerate(envs):
        new_env.update_distribution_from_other(env)
        new_env.seed(i)
        new_env.set_task()
        new_env.reset()
    algo = PPOAlgo(policy, envs, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                   args.gae_lambda,
                   args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                   args.optim_eps, args.clip_eps, args.epochs, args.meta_batch_size,
                   parallel=not args.sequential, rollouts_per_meta_task=args.rollouts_per_meta_task,
                   obs_preprocessor=obs_preprocessor, instr_dropout_prob=args.instr_dropout_prob)
    if 'optimizer' in model_data:
        algo.optimizer.load_state_dict(model_data['optimizer'])

    teacher_schedule = make_teacher_schedule(args.feedback_type, args.teacher_schedule)
    # Standardize args
    args.single_level = True
    args.reward_when_necessary = False  # TODO: make this a flag

    finetune_sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=args.rollouts_per_meta_task,
        meta_batch_size=num_rollouts,
        max_path_length=args.max_path_length,
        parallel=False,
        envs_per_task=1,
        reward_predictor=None,
        supervised_model=supervised_model,
        obs_preprocessor=obs_preprocessor,
    )

    def log_fn_vidrollout(rl_policy, il_policy, itr):
        policy = rl_policy if il_policy is None else il_policy
        test_success_checkpoint(heldout_env, save_dir, 3, teachers, policy=policy, policy_name=policy_name,
                                env_name=env_name, hide_instrs=hide_instrs, itr=itr, stochastic=stochastic)

    def log_fn(rl_policy, il_policy, logger, itr):
        if itr % 10 == 0:
            log_fn_vidrollout(rl_policy, il_policy, itr)
        policy_env_name = f'Policy{policy_name}-{env_name}'
        full_save_dir = save_dir.joinpath(policy_env_name + '_checkpoint')
        if itr == 0:
            if not full_save_dir.exists():
                full_save_dir.mkdir()
            with open(full_save_dir.joinpath('results.csv'), 'w') as f:
                f.write('policy_env,policy,env,success_rate,stoch_accuracy,itr \n')
        policy = il_policy if il_policy is not None else rl_policy
        teacher_dict = {k: k in teachers for k, v in teacher_null_dict.items()}
        seeds = np.arange(1000, 1000 + finetune_sampler.meta_batch_size)
        finetune_sampler.vec_env.seed(seeds)
        finetune_sampler.vec_env.set_tasks()
        paths = finetune_sampler.obtain_samples(log=False, advance_curriculum=False, policy=policy,
                                                teacher_dict=teacher_dict, max_action=False, show_instrs=not hide_instrs)
        data = sample_processor.process_samples(paths, log_prefix='n/a', log_teacher=False)

        num_total_episodes = data['dones'].sum()
        num_successes = data['env_infos']['success'].sum()
        avg_success = num_successes / num_total_episodes
        # Episode length contains the timestep, starting at 1.  Padding values are 0.
        pad_steps = (data['env_infos']['episode_length'] == 0).sum()
        correct_actions = (data['actions'] == data['env_infos']['teacher_action'][:, :, 0]).sum() - pad_steps
        avg_accuracy = correct_actions / (np.prod(data['actions'].shape) - pad_steps)
        print(f"Finetuning achieved success: {avg_success}, stoch acc: {avg_accuracy}")
        with open(full_save_dir.joinpath('results.csv'), 'a') as f:
            f.write(
                f'{policy_env_name},{policy_name},{env_name},{avg_success},{avg_accuracy},{itr} \n')
        return avg_success, avg_accuracy

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
        buffer_name=save_name,
        exp_name=save_name,
        curriculum_step=env_index,
        il_trainer=il_trainer,
        supervised_model=supervised_model,
        reward_predictor=None,
        rp_trainer=rp_trainer,
        is_debug=False,
        teacher_schedule=teacher_schedule,
        obs_preprocessor=obs_preprocessor,
        log_dict={},
        log_and_save=True,#False,
        eval_heldout=False,
        log_fn=log_fn,
        log_every=1,
    )
    trainer.train()
    print("All done!")


def test_success(env, env_index, save_dir, num_rollouts, teachers, teacher_null_dict, policy_path=None, policy=None,
                 policy_name="", env_name="", hide_instrs=False, heldout_env=[], stochastic=True, additional_args={}):
    if policy is None:
        policy, il_model, _, args, model_data = load_policy(policy_path)
        for k, v in additional_args.items():
            setattr(args, k, v)
        n_itr = args.n_itr
    else:
        n_itr = 0
    policy_env_name = f'Policy{policy_name}-{env_name}'
    print("EVALUATING", policy_env_name)
    full_save_dir = save_dir.joinpath(policy_env_name)
    if not full_save_dir.exists():
        full_save_dir.mkdir()
    if n_itr > 0:
        finetune_path = full_save_dir.joinpath('finetuned_policy')
        if not finetune_path.exists():
            finetune_path.mkdir()
        finetune_policy(env, env_index, policy, il_model,
                        finetune_path, args, teacher_null_dict,
                        save_dir=save_dir, teachers=teachers, policy_name=policy_name, env_name=env_name,
                        hide_instrs=hide_instrs, heldout_env=heldout_env, stochastic=stochastic,
                        num_rollouts=num_rollouts, model_data=model_data)
    success_rate, stoch_accuracy, det_accuracy, followed_cc3 = eval_policy(env, il_model, full_save_dir, num_rollouts,
                                                                           teachers, hide_instrs, stochastic)
    print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    with open(save_dir.joinpath('results.csv'), 'a') as f:
        f.write(
            f'{policy_env_name},{policy_name},{env_name},{success_rate},{stoch_accuracy},{det_accuracy},{followed_cc3} \n')
    return success_rate, stoch_accuracy, det_accuracy


def test_success_checkpoint(env, save_dir, num_rollouts, teachers, policy=None,
                            policy_name="", env_name="", hide_instrs=False, itr=-1, stochastic=True):
    policy_env_name = f'Policy{policy_name}-{env_name}'
    full_save_dir = save_dir.joinpath(policy_env_name + '_checkpoint')
    if not full_save_dir.exists():
        full_save_dir.mkdir()
    success_rate, stoch_accuracy, det_accuracy, followed_cc3 = eval_policy(env, policy, full_save_dir, num_rollouts,
                                                                           teachers, hide_instrs, stochastic,
                                                                           f'vid_{itr}')
    print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    return success_rate, stoch_accuracy, det_accuracy


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument('--envs', nargs='+', required=True, type=str)
    parser.add_argument('--levels', nargs='+', default=['latest'], type=str)
    parser.add_argument('--teachers', nargs='+', default=['all'], type=str)
    parser.add_argument("--finetune_itrs", default=0, type=int)
    parser.add_argument("--num_rollouts", default=50, type=int)
    parser.add_argument("--no_train_rl", action='store_true')
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--hide_instrs", action='store_true')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument('--teacher_schedule', type=str, default='all_teachers')
    parser.add_argument('--distillation_strategy', type=str, choices=[
            'all_teachers', 'no_teachers', 'all_but_none', 'powerset'
        ], default='distill_powerset')
    parser.add_argument('--no_distill', action='store_true')
    parser.add_argument('--yes_distill', action='store_true')
    parser.add_argument('--rollout_temperature', type=float, default=1)
    parser.add_argument('--finetune_il', action='store_true')
    args = parser.parse_args()

    save_dir = pathlib.Path(args.save_dir)
    policy_path = pathlib.Path(args.policy)

    _, _, default_env, config, model_data = load_policy(policy_path.joinpath('latest.pkl'))
    default_env.reset()
    teacher_null_dict = default_env.teacher.null_feedback()

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
            index = int(env_name[4])
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
    additional_args['yes_distill'] = args.yes_distill
    additional_args['no_distill'] = args.no_distill
    additional_args['rollout_temperature'] = args.rollout_temperature
    additional_args['finetune_il'] = args.finetune_il

    # TODO: eventually remove!
    additional_args['distill_successful_only'] = False
    additional_args['min_itr_steps_distill'] = 0

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
            test_success(env, env_index, save_dir, args.num_rollouts, args.teachers, teacher_null_dict,
                         policy_path=policy_path.joinpath(policy_name),
                         policy_name=policy_path.stem, env_name=inner_env.__class__.__name__,
                         hide_instrs=args.hide_instrs, heldout_env=env, stochastic=not args.deterministic,
                         additional_args=additional_args)


if __name__ == '__main__':
    main()
