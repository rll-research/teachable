import numpy as np
import argparse
import sys
import joblib
import tensorflow as tf
import time
from meta_mb.logger import logger
from meta_mb.envs.normalized_env import normalize
from meta_mb.meta_envs.mujoco.ant_rand_goal import AntRandGoalEnv
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.samplers.meta_samplers import MAMLSampler
from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
from meta_mb.meta_algos.vpg_maml import VPGMAML

BATCH_SIZE = 80
META_BATCH_SIZE = 40
PATH_LENGTH = 200
NUM_INNER_GRAD_STEPS = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default=None,
                        help='policy to load')
    args = parser.parse_args(sys.argv[1:])

    sess = tf.InteractiveSession()

    policy = joblib.load(args.policy)['policy']
    policy.switch_to_pre_update()

    baseline = LinearFeatureBaseline()

    env = normalize(AntRandGoalEnv())

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=BATCH_SIZE,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=PATH_LENGTH,
        parallel=True,
        envs_per_task=20,
    )

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=0.99,
        gae_lambda=1,
        normalize_adv=True,
        positive_adv=False,
    )

    # Doesn't matter which algo
    algo = VPGMAML(
        policy=policy,
        inner_lr=0.1,
        meta_batch_size=META_BATCH_SIZE,
        inner_type='likelihood_ratio',
        num_inner_grad_steps=NUM_INNER_GRAD_STEPS,
    )

    uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(uninit_vars))
    
    # Preupdate:
    tasks = env.sample_tasks(META_BATCH_SIZE)
    sampler.vec_env.set_tasks(tasks)
    
    # Preupdate:
    for i in range(NUM_INNER_GRAD_STEPS):
        paths = sampler.obtain_samples(log=False)
        samples_data = sample_processor.process_samples(paths, log=True, log_prefix='%i_' % i)
        env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % i)
        algo._adapt(samples_data)

    paths = sampler.obtain_samples(log=False)
    samples_data = sample_processor.process_samples(paths, log=True, log_prefix='%i_' % NUM_INNER_GRAD_STEPS)
    env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % NUM_INNER_GRAD_STEPS)
    logger.dumpkvs()

    # Postupdate:
    while True:
        task_i = np.random.choice(range(META_BATCH_SIZE))
        env.set_task(tasks[task_i])
        print(tasks[task_i])
        obs = env.reset()
        for _ in range(PATH_LENGTH):
            env.render()
            action, _ = policy.get_action(obs, task_i)
            obs, reward, done, _ = env.step(action)
            time.sleep(0.001)
            if done:
                break