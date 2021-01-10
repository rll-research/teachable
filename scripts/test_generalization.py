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
    try:
        policy.instr_rnn.flatten_parameters()
    except Exception as e:
        print(e, "looks like instrs aren't rnn")
    return policy, env, args


def eval_policy(env, policy, save_dir, num_rollouts, teachers, hide_instrs, stochastic):
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
    paths, accuracy, stoch_accuracy, det_accuracy, followed_cc3 = rollout(env, policy,
                                                            instrs=not hide_instrs,
                                                            reset_every=1,
                                                            stochastic=stochastic,
                                                            record_teacher=True,
                                                            teacher_dict=teacher_dict,
                                                            video_directory=save_dir,
                                                            video_name='generalization_vids',
                                                            num_rollouts=num_rollouts,
                                                            save_wandb=False,
                                                            save_locally=True,
                                                            num_save=num_rollouts,
                                                            obs_preprocessor=obs_preprocessor,
                                                            rollout_oracle=False)
print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    with open(save_dir.joinpath('results.csv'), 'a') as f:
        f.write(
            f'{policy_env_name},{policy_name},{env_name},{success_rate},{stoch_accuracy},{det_accuracy},{followed_cc3} \n')
    return success_rate, stoch_accuracy, det_accuracy


def test_success_checkpoint(env, save_dir, num_rollouts, teachers, policy=None,
                            policy_name="", env_name="", hide_instrs=False, itr=-1):
    policy_env_name = f'Policy{policy_name}-{env_name}'
    full_save_dir = save_dir.joinpath(policy_env_name + '_checkpoint')
    print("FSD", full_save_dir)
    if not full_save_dir.exists():
        full_save_dir.mkdir()
    success_rate, stoch_accuracy, det_accuracy, followed_cc3 = eval_policy(env, policy, full_save_dir, num_rollouts,
                                                                           teachers, hide_instrs)
    print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    with open(full_save_dir.joinpath('results.csv'), 'a') as f:
        f.write(
            f'{policy_env_name},{policy_name},{env_name},{success_rate},{stoch_accuracy},{det_accuracy},{followed_cc3}, {itr} \n')
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
    parser.add_argument("--hide_instrs", action='store_true')
    parser.add_argument("--deterministic", action='store_true')
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
        f.write('policy_env,policy, env,success_rate, stoch_accuracy, det_accuracy, followed_cc3 \n')
    for policy_name in policy_level_names:
        for env in envs:
            test_success(env, save_dir, args.finetune_itrs,
                         args.num_rollouts, args.teachers, teacher_null_dict,
                         policy_path=policy_path.joinpath(policy_name),
                         policy_name=policy_path.stem, env_name=env.__class__.__name__, 
                         hide_instrs=args.hide_instrs, heldout_envs=envs, stochastic=not args.deterministic)


if __name__ == '__main__':
    main()
