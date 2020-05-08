from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.meta_samplers.meta_vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from meta_mb.logger import logger
from meta_mb.utils import utils
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools


class SingleMetaSampler(BaseSampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_mb.meta_envs.base.MetaEnv) : environment object
        policy (maml_zoo.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of meta_envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            policy,
            rollouts_per_meta_task,
            meta_batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=False,

            ):
        super(SingleMetaSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.samples_per_task = rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0

    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        tasks = self.env.sample_tasks(self.meta_batch_size)
        assert len(tasks) == self.meta_batch_size
        self.env.set_tasks(tasks)

    def set_tasks(self, tasks):
        assert len(tasks) == self.meta_batch_size
        self.env.set_tasks(tasks)

    def obtain_samples(self, log=False, log_prefix='', random=False):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random

        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        running_paths = _get_empty_running_paths_dict()

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        for idx in range(self.meta_batch_size):
            ts = 0
            n_samples = 0

            init_obs = np.expand_dims(self.env.reset(), 0).copy()
            obses = [init_obs for _ in range(self.meta_batch_size)]
            policy.reset(dones=[True] * self.meta_batch_size)
            while n_samples < self.samples_per_task:
                # execute policy
                t = time.time()

                if random:
                    actions = np.stack([[self.env.action_space.sample()] for _ in range(len(obses))], axis=0)
                    agent_infos = [[{'mean': np.zeros_like(self.env.action_space.sample()),
                                     'log_std': np.zeros_like(self.env.action_space.sample())}] * self.envs_per_task] * self.meta_batch_size
                else:
                    actions, agent_infos = policy.get_actions(obses)

                policy_time += time.time() - t

                # step environments
                t = time.time()
                action, agent_info = actions[idx][0], agent_infos[idx][0]
                observation = obses[idx][0].copy()

                next_obs, reward, done, env_info = self.env.step(action)

                ts += 1
                done = done or ts >= self.max_path_length
                if done:
                    next_obs = self.env.reset()
                    # time.sleep(1)
                    ts = 0

                env_time += time.time() - t

                new_samples = 0
                # append new samples to running paths
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                running_paths["observations"].append(observation)
                running_paths["actions"].append(action)
                running_paths["rewards"].append(reward)
                running_paths["dones"].append(done)
                running_paths["env_infos"].append(env_info)
                running_paths["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths[idx].append(dict(
                        observations=np.asarray(running_paths["observations"]),
                        actions=np.asarray(running_paths["actions"]),
                        rewards=np.asarray(running_paths["rewards"]),
                        dones=np.asarray(running_paths["dones"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths["agent_infos"]),
                    ))
                    new_samples += len(running_paths["rewards"])
                    running_paths = _get_empty_running_paths_dict()

                pbar.update(new_samples)
                n_samples += new_samples
                obses[idx][0] = next_obs

            self.total_timesteps_sampled += n_samples

        pbar.stop()
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
