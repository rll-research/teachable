from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.metrpo_samplers.metrpo_env_executor import METRPOIterativeEnvExecutor
from meta_mb.logger import logger
from meta_mb.utils import utils
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools


class METRPOSampler(BaseSampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_mb.meta_envs.base.MetaEnv) : environment object
        policy (meta_mb.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of meta_envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            policy,
            dynamics_model,
            num_rollouts,
            max_path_length,
            parallel=False,
            deterministic=True,
            ):
        super(METRPOSampler, self).__init__(env, policy, num_rollouts, max_path_length)
        assert not parallel

        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length

        self.total_samples = num_rollouts * max_path_length
        self.total_timesteps_sampled = 0

        # setup vectorized environment
        self.vec_env = METRPOIterativeEnvExecutor(env, dynamics_model, num_rollouts, max_path_length,
                                                  deterministic=deterministic)

    def obtain_samples(self, log=False, log_prefix='', buffer=None):
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
        paths = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        # pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy
        policy.reset(dones=[True] * self.vec_env.num_envs)

        # initial reset of meta_envs
        obses = np.asarray(self.vec_env.reset(buffer))

        while n_samples < self.total_samples:

            # execute policy
            t = time.time()
            actions, agent_infos = policy.get_actions(obses)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses,
                                                                                    actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["dones"].append(done)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths.append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        dones=np.asarray(running_paths[idx]["dones"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            # pbar.update(self.vec_env.num_envs)
            n_samples += new_samples
            obses = next_obses
        # pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        logger.logkv('ModelSampler-n_timesteps', self.total_timesteps_sampled)
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        return agent_infos, env_infos


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
