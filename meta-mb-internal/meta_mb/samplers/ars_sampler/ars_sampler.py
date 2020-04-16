from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.ars_sampler.ars_env_executor import ModelARSEnvExecutor, IterativeARSEnvExecutor
from meta_mb.samplers.vectorized_env_executor import IterativeEnvExecutor, ParallelEnvExecutor
from meta_mb.logger import logger
from meta_mb.utils import utils
from collections import OrderedDict
from pyprind import ProgBar
import numpy as np
import time
import itertools


class ARSSampler(BaseSampler):
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
            rollouts_per_policy,
            num_deltas,
            max_path_length,
            dynamics_model=None,
            n_parallel=1,
            vae=None,
            ):
        super(ARSSampler, self).__init__(env, policy, rollouts_per_policy, max_path_length)

        self.rollouts_per_policy = rollouts_per_policy
        self.num_deltas = num_deltas
        self.total_samples = num_deltas * rollouts_per_policy * max_path_length * 2
        self.total_timesteps_sampled = 0
        self.dynamics_model = dynamics_model
        self.vae = vae

        # setup vectorized environment
        # TODO: Create another vectorized env executor
        if dynamics_model is not None:
            self.vec_env = ModelARSEnvExecutor(env, dynamics_model, num_deltas, rollouts_per_policy,
                                          max_path_length)
        else:
            if n_parallel > 1:
                self.vec_env = ParallelEnvExecutor(env, n_parallel, 2 * rollouts_per_policy * num_deltas, max_path_length)
            else:
                self.vec_env = IterativeARSEnvExecutor(env, num_deltas, rollouts_per_policy, max_path_length)

    def obtain_samples(self, log=False, log_prefix='', buffer=None):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        pbar = ProgBar(self.max_path_length)
        policy_time, env_time = 0, 0

        policy = self.policy
        policy.reset(dones=[True] * self.vec_env.num_envs)

        # initial reset of meta_envs
        obses = self.vec_env.reset(buffer)
        time_step = 0
        list_observations = []
        list_actions = []
        list_rewards = []
        list_dones = []
        mask = np.ones((self.vec_env.num_envs,))

        while time_step < self.max_path_length:
            
            # Execute policy
            t = time.time()
            if self.vae is not None:
                obses = np.array(obses)
                obses = self.vae.encode(obses)
                obses = np.split(obses, self.vec_env.num_envs, axis=0)
            if self.dynamics_model is not None:
                actions, agent_infos = policy.get_actions_batch(obses, update_filter=False)
            else:
                obses = np.array(obses)
                actions, agent_infos = policy.get_actions_batch(obses, update_filter=True)
            policy_time += time.time() - t

            # Step environments
            t = time.time()
            next_obses, rewards, dones, _ = self.vec_env.step(actions)
            next_obses, rewards, dones = np.array(next_obses), np.array(rewards), np.array(dones)

            rewards *= mask
            dones = dones + (1 - mask)
            mask *= (1 - dones)

            env_time += time.time() - t

            list_observations.append(obses)
            list_actions.append(actions)
            list_rewards.append(rewards)
            list_dones.append(dones)

            time_step += 1
            obses = next_obses
            pbar.update(1)
        pbar.stop()
        self.total_timesteps_sampled += np.sum(1 - np.array(list_dones))

        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        samples_data = dict(observations=np.array(list_observations),
                            actions=np.array(list_actions),
                            rewards=np.array(list_rewards),
                            returns=np.sum(list_rewards, axis=0),
                            dones=np.array(list_dones))

        return samples_data
