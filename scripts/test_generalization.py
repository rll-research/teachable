import numpy as np
import pathlib
import time

from utils.rollout import rollout
from envs.babyai.utils.obs_preprocessor import make_obs_preprocessor
import matplotlib
matplotlib.use('Agg')


def eval_policy(env, policy, save_dir, num_rollouts, hide_instrs, stochastic, args, seed=0,
                video_name='generalization_vids', num_save=20, feedback_list=[]):
    if not save_dir.exists():
        save_dir.mkdir()
    env.seed(seed)
    env.reset()
    obs_preprocessor = make_obs_preprocessor(feedback_list=feedback_list)
    policy.train(False)
    paths, accuracy, stoch_accuracy, det_accuracy, reward = rollout(env, policy,
                                                                    instrs=not hide_instrs,
                                                                    reset_every=1,
                                                                    stochastic=stochastic,
                                                                    record_teacher=True,
                                                                    video_directory=save_dir,
                                                                    video_name=video_name,
                                                                    num_rollouts=num_rollouts,
                                                                    save_wandb=False,
                                                                    save_locally=num_save > 0,
                                                                    num_save=num_save,
                                                                    obs_preprocessor=obs_preprocessor,
                                                                    rollout_oracle=False,
                                                                    hierarchical_rollout=args.algo=='hppo')
    success_rate = np.mean([path['env_infos'][-1]['success'] for path in paths])
    try:
        success_rate = np.mean([path['env_infos'][-1]['timestep_success'] for path in paths])
    except:
        print("doesn't have timestep_success")
    return success_rate, stoch_accuracy, det_accuracy, reward


def make_log_fn(env, args, start_num_feedback, save_dir, policy, hide_instrs, seed=1, stochastic=True,
                num_rollouts=10, policy_name='policy', env_name='env', log_every=10, feedback_list=[]):
    start = time.time()
    save_dir = pathlib.Path(save_dir)

    def log_fn_vidrollout(itr, num_save):
        return test_success_checkpoint(env, save_dir, num_rollouts, policy=policy, policy_name=policy_name,
                                       env_name=env_name, hide_instrs=hide_instrs, itr=itr, stochastic=stochastic,
                                       args=args,
                                       seed=seed, num_save=num_save, feedback_list=feedback_list)

    def log_fn(itr, num_feedback):
        policy_env_name = f'Policy{policy_name}-{env_name}'
        full_save_dir = save_dir.joinpath(policy_env_name + f'_checkpoint{seed}')
        if itr == 0:
            if not full_save_dir.exists():
                full_save_dir.mkdir()
            file_name = full_save_dir.joinpath('results.csv')
            if not file_name.exists():
                with open(file_name, 'w') as f:
                    f.write('policy_env,policy,env,success_rate,stoch_accuracy,itr,num_feedback,time,reward\n')
        if not itr % log_every == 0:
            return
        num_save = 10
        avg_success, avg_accuracy, det_accuracy, reward = log_fn_vidrollout(itr, num_save)
        print(f"Finetuning achieved success: {avg_success}, stoch acc: {avg_accuracy}")
        with open(full_save_dir.joinpath('results.csv'), 'a') as f:
            f.write(
                f'{policy_env_name},{policy_name},{env_name},{avg_success},{avg_accuracy},{itr},'
                f'{num_feedback + start_num_feedback},{time.time() - start},{reward} \n')
        return avg_success, avg_accuracy

    return log_fn

def test_success_checkpoint(env, save_dir, num_rollouts, policy=None,
                            policy_name="", env_name="", hide_instrs=False, itr=-1, stochastic=True, args=None,
                            seed=0, num_save=0, feedback_list=[]):
    policy_env_name = f'Policy{policy_name}-{env_name}'
    full_save_dir = save_dir.joinpath(policy_env_name + f'_checkpoint{seed}')
    if not full_save_dir.exists():
        full_save_dir.mkdir()
    success_rate, stoch_accuracy, det_accuracy, reward = eval_policy(env, policy, full_save_dir, num_rollouts,
                                                                     hide_instrs, stochastic, args,
                                                                     seed, f'vid', num_save=num_save,
                                                                     feedback_list=feedback_list)
    print(f"Finished with success: {success_rate}, stoch acc: {stoch_accuracy}, det acc: {det_accuracy}")
    return success_rate, stoch_accuracy, det_accuracy, reward