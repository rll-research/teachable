# imports
import joblib
import os
import numpy as np
import argparse
import pathlib

from meta_mb.samplers.utils import rollout
# from meta_mb.trainers.mf_trainer import Trainer
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
                                                            max_path_length=200,
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


def finetune_policy(env, policy, finetuning_epochs, config, save_name):
    # save_name = "ORACLE" + save_name
    policy._hidden_state = None
    reward_predictor._hidden_state = None
    sample_processor = RL2SampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )
    algo = PPO(
        policy=policy,
        learning_rate=config['learning_rate'],
        max_epochs=config['max_epochs'],
        backprop_steps=config['backprop_steps'],
        reward_predictor=reward_predictor,
        entropy_bonus=config['entropy_bonus'],
    )
    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=1,
        reward_predictor=reward_predictor,
    )
    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=finetuning_epochs,
        sess=sess,
        start_itr=0,
        reward_threshold=2,
        config=config,
        log_and_save=False,
        use_rp_inner=True,
        use_rp_outer=True,
    )
    trainer.train()
    print("Saving policy", save_name)
    save_file(policy, save_name)
    print("All done!")


def test_success(policy_path, env, save_dir, finetune_itrs, config, num_rollouts, teachers):
    policy, _, _ = load_policy(policy_path)
    policy_env_name = f'Policy{policy_path.stem}-{env.__class__.__name__}'
    print("EVALUATING", policy_env_name)
    full_save_dir = save_dir.joinpath(policy_env_name)
    if finetune_itrs > 0:
        finetune_policy(env, policy, finetune_itrs, config, full_save_dir.joinpath('finetuned_policy.pt'))
    success_rate, stoch_accuracy, det_accuracy = eval_policy(env, policy, full_save_dir, num_rollouts, teachers)
    print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    with open(save_dir.joinpath('results.csv'), 'a') as f:
        f.write(f'{policy_env_name},{policy_path.stem},{env.__class__.__name__},{success_rate},{stoch_accuracy},{det_accuracy} \n')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument('--envs', nargs='+', required=True, type=str)
    parser.add_argument('--levels', nargs='+', default=['latest'], type=str)
    parser.add_argument('--teachers', nargs='+', default=['all'], type=str)
    parser.add_argument("--finetune_itrs", default=0, type=int)
    parser.add_argument("--num_rollouts", default=50, type=int)
    parser.add_argument("--save_dir", default=".")
    args = parser.parse_args()

    save_dir = pathlib.Path(args.save_dir)
    policy_path = pathlib.Path(args.policy)

    _, default_env, config = load_policy(policy_path.joinpath('latest.pkl'))

    # Get the levels of the policies to load
    policy_levels = args.levels
    if policy_levels == ['all']:
        policy_levels = range(len(default_env.levels_list))
    policy_level_names = []
    for policy_level in policy_levels:
        try:
            level_number = int(policy_level)
            policy_level_names.append(f'level{level_number}.pkl')
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
            test_success(policy_path.joinpath(policy_name), env, save_dir, args.finetune_itrs, config,
                         args.num_rollouts, args.teachers)


if __name__ == '__main__':
    main()
