#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import gym
import logging
import sys
import subprocess
import os
import time
import numpy as np
import blosc
import torch
import copy

import babyai.utils as utils

# Parse arguments

# Set seed for all randomness sources


def generate_demos(env, buffer, n_episodes):
    demos = []
    for i in range(n_episodes):
        if not i % 100:
            print("Demo", i)
        demo = generate_demo(env)
        demos.append(demo)
    buffer.add_trajs_no_split(demos, 18)


def generate_demo(env):
    # utils.seed(seed)

    agent = utils.load_agent(env, 'BOT', 'DUMMY', 'agent', False, 'whatever')
    demos = []

    obs = env.reset()
    instr = obs['instr']
    obs = {'image': obs['obs'], 'direction': 1, 'mission': env.mission}
    agent.on_reset()

    actions = []
    images = []
    directions = []
    done = False

    try:
        while not done:
            # action = agent.act(obs)['action']
            action = env.get_teacher_action()
            if isinstance(action, torch.Tensor):
                action = action.item()
            new_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)

            actions.append(action)
            images.append(obs['image'])
            directions.append(obs['direction'])

            obs = {'image': new_obs['obs'], 'direction': 1, 'mission': env.mission}
        if reward == 0:
            raise Exception("mission failed, the seed is {}".format(-1))
    except (Exception, AssertionError):
        raise

    def one_hotify(a):
        pre_action_advice = np.zeros(8)
        pre_action_advice[a] = 1
        return pre_action_advice

    done_vec = np.zeros(len(actions))
    done_vec[-1] = 0
    obs = [{'obs': o, 'instr': copy.deepcopy(instr), 'PreActionAdvice': one_hotify(a)} for o, a in zip(images, actions)]
    demo = {'obs': obs, 'action': np.concatenate(actions), 'teacher_action': np.concatenate(actions), 'full_done': done_vec}

    return demo


# def generate_demos_cluster():
#     demos_per_job = args.episodes // args.jobs
#     demos_path = utils.get_demos_path(args.demos, args.env, 'agent')
#     job_demo_names = [os.path.realpath(demos_path + '.shard{}'.format(i))
#                      for i in range(args.jobs)]
#     for demo_name in job_demo_names:
#         job_demos_path = utils.get_demos_path(demo_name)
#         if os.path.exists(job_demos_path):
#             os.remove(job_demos_path)
#
#     command = [args.job_script]
#     command += sys.argv[1:]
#     for i in range(args.jobs):
#         cmd_i = list(map(str,
#             command
#               + ['--seed', args.seed + i * demos_per_job]
#               + ['--demos', job_demo_names[i]]
#               + ['--episodes', demos_per_job]
#               + ['--jobs', 0]
#               + ['--valid-episodes', 0]))
#         logger.info('LAUNCH COMMAND')
#         logger.info(cmd_i)
#         output = subprocess.check_output(cmd_i)
#         logger.info('LAUNCH OUTPUT')
#         logger.info(output.decode('utf-8'))
#
#     job_demos = [None] * args.jobs
#     while True:
#         jobs_done = 0
#         for i in range(args.jobs):
#             if job_demos[i] is None or len(job_demos[i]) < demos_per_job:
#                 try:
#                     logger.info("Trying to load shard {}".format(i))
#                     job_demos[i] = utils.load_demos(utils.get_demos_path(job_demo_names[i]))
#                     logger.info("{} demos ready in shard {}".format(
#                         len(job_demos[i]), i))
#                 except Exception:
#                     logger.exception("Failed to load the shard")
#             if job_demos[i] and len(job_demos[i]) == demos_per_job:
#                 jobs_done += 1
#         logger.info("{} out of {} shards done".format(jobs_done, args.jobs))
#         if jobs_done == args.jobs:
#             break
#         logger.info("sleep for 60 seconds")
#         time.sleep(60)
#
#     # Training demos
#     all_demos = []
#     for demos in job_demos:
#         all_demos.extend(demos)
#     utils.save_demos(all_demos, demos_path)

