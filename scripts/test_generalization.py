# imports
import joblib
import os
import copy
import numpy as np
import argparse
import pathlib

from meta_mb.samplers.utils import rollout
from babyai.utils.obs_preprocessor import make_obs_preprocessor


def load_policy(path):
    saved_model = joblib.load(path)
    env = saved_model['env']
    policy = saved_model['policy']
    args = saved_model['args']
    return policy, env, args


def eval_policy(env, policy, save_dir, num_rollouts, teachers):
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
    paths, accuracy, stoch_accuracy, det_accuracy = rollout(env, policy,
                                                            reset_every=1,
                                                            stochastic=True,
                                                            record_teacher=True,
                                                            teacher_dict=teacher_dict,
                                                            video_directory=save_dir,
                                                            video_name='generalization_vids',
                                                            num_rollouts=num_rollouts,
                                                            save_wandb=False,
                                                            save_locally=True,
                                                            num_save=5,
                                                            obs_preprocessor=obs_preprocessor,
                                                            rollout_oracle=False)
    success_rate = np.mean([path['env_infos'][-1]['success'] for path in paths])
    return success_rate, stoch_accuracy, det_accuracy


def finetune_policy(env, policy, supervised_model, finetuning_epochs, save_name, args, teacher_null_dict):
    from meta_mb.algos.ppo_torch import PPOAlgo
    from meta_mb.trainers.mf_trainer import Trainer
    from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
    from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
    from meta_mb.trainers.il_trainer import ImitationLearning
    from babyai.teacher_schedule import make_teacher_schedule

    print("TEACHER NULL DICT", teacher_null_dict)
    obs_preprocessor = make_obs_preprocessor(teacher_null_dict)

    args.model = 'default_il'
    args.instr_dropout_prob = 0 # TODO move all these together
    if supervised_model is not None:
        il_trainer = ImitationLearning(supervised_model, env, args, distill_with_teacher=False,
                                       preprocess_obs=obs_preprocessor, label_weightings=args.distill_label_weightings,
                                       instr_dropout_prob=args.instr_dropout_prob)
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
    algo = PPOAlgo(policy, envs, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                   args.gae_lambda,
                   args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                   args.optim_eps, args.clip_eps, args.epochs, args.meta_batch_size,
                   parallel=not args.sequential, rollouts_per_meta_task=args.rollouts_per_meta_task,
                   obs_preprocessor=obs_preprocessor)

    teacher_schedule = make_teacher_schedule(args.feedback_type, 'last_teacher')
    # Standardize args
    args.single_level = True
    args.n_itr = finetuning_epochs
    args.instr_dropout_prob = 0 # TODO: ??
    args.reward_when_necessary = False  # TODO: make this a flag

    trainer = Trainer(
        args,
        algo=algo,
        policy=policy,
        env=copy.deepcopy(env),
        sampler=sampler,
        sample_processor=sample_processor,
        buffer_name=save_name.parent,
        exp_name=save_name,
        curriculum_step=0,
        il_trainer=il_trainer,
        supervised_model=supervised_model,
        reward_predictor=None,
        rp_trainer=rp_trainer,
        is_debug=False,
        teacher_schedule=teacher_schedule,
        obs_preprocessor=obs_preprocessor,
        log_dict={},
        log_and_save=False,
        eval_heldout=False,
    )
    trainer.train()  # TODO: add this!
    # print("Saving policy", save_name)
    # save_file(policy, save_name)
    print("All done!")


def test_success(env, save_dir, finetune_itrs, num_rollouts, teachers, teacher_null_dict,
                 policy_path=None, policy=None,
                 policy_name="", env_name=""):
    if policy is None:
        policy, _, args = load_policy(policy_path)
    policy_env_name = f'Policy{policy_name}-{env_name}'
    print("EVALUATING", policy_env_name)
    full_save_dir = save_dir.joinpath(policy_env_name)
    if not full_save_dir.exists():
        full_save_dir.mkdir()
    if finetune_itrs > 0:
        finetune_policy(env, policy, policy, finetune_itrs, full_save_dir.joinpath('finetuned_policy.pt'), args, teacher_null_dict)
    success_rate, stoch_accuracy, det_accuracy = eval_policy(env, policy, full_save_dir, num_rollouts, teachers)
    print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    with open(save_dir.joinpath('results.csv'), 'a') as f:
        f.write(f'{policy_env_name},{policy_name},{env_name},{success_rate},{stoch_accuracy},{det_accuracy} \n')
    return success_rate, stoch_accuracy, det_accuracy


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument('--envs', nargs='+', required=True, type=str)
    parser.add_argument('--levels', nargs='+', default=['latest'], type=str)
    parser.add_argument('--teachers', nargs='+', default=['all'], type=str)
    parser.add_argument("--finetune_itrs", default=0, type=int)
    parser.add_argument("--num_rollouts", default=50, type=int)
    parser.add_argument("--train_rl_on_finetune", action='store_true')
    parser.add_argument("--save_dir", default=".")
    args = parser.parse_args()

    save_dir = pathlib.Path(args.save_dir)
    policy_path = pathlib.Path(args.policy)

    _, default_env, config = load_policy(policy_path.joinpath('latest.pkl'))
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
    envs = []
    for env_name in env_names:
        if env_name == 'train':
            envs += default_env.levels_list
        elif env_name == 'test':
            envs += default_env.held_out_levels
        else:
            try:
                env_id = int(env_name)
                envs.append(default_env.levels_list[env_id])
            except ValueError:
                for level in default_env.levels_list:
                    if env_name in level.__class__.__name__:
                        envs.append(level)

    # Test every policy with every level
    if not save_dir.exists():
        save_dir.mkdir()
    with open(save_dir.joinpath('results.csv'), 'w') as f:
        f.write('policy_env,policy, env,success_rate, stoch_accuracy, det_accuracy \n')
    for policy_name in policy_level_names:
        for env in envs:
            test_success(env, save_dir, args.finetune_itrs,
                         args.num_rollouts, args.teachers, teacher_null_dict,
                         policy_path=policy_path.joinpath(policy_name),
                         policy_name=policy_path.stem, env_name=env.__class__.__name__)


if __name__ == '__main__':
    main()
