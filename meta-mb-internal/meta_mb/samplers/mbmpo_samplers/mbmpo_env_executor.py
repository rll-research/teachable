import numpy as np


class MBMPOIterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.

    Args:
        env (meta_mb.meta_envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                               the respective environment is reset
    """

    def __init__(self, env, dynamics_model, meta_batch_size, envs_per_task, max_path_length, deterministic=True):
        self.env = env
        self.dynamics_model = dynamics_model
        self._num_envs = meta_batch_size * envs_per_task

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        # check whether env has done function
        self.has_done_fn = hasattr(self.unwrapped_env, 'done')

        self.ts = np.zeros(meta_batch_size * envs_per_task, dtype='int')  # time steps
        self.max_path_length = max_path_length
        self.current_obs = None
        self._buffer = None

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of meta_envs)
        """
        assert len(actions) == self.num_envs
        prev_obs = self.current_obs
        next_obs = self.dynamics_model.predict_batches(prev_obs, actions)

        rewards = self.unwrapped_env.reward(prev_obs, actions, next_obs)

        if self.has_done_fn:
            dones = self.unwrapped_env.done(next_obs)
        else:
            dones = np.asarray([False for _ in range(self.num_envs)])

        env_infos = [{} for _ in range(self.num_envs)]

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)
        for i in np.argwhere(dones).flatten():
            next_obs[i], self.ts[i] = self._reset()

        self.current_obs = next_obs

        return next_obs, rewards, dones, env_infos

    def set_tasks(self, tasks):
        """
        Sets a list of tasks to each environment

        Args:
            tasks (list): list of the tasks for each environment
        """
        pass

    def _reset(self):
        if self._buffer is None:
            return self.env.reset(), 0

        else:
            initial_idxs = np.arange(len(self._buffer['time_steps']))[self._buffer['time_steps'] == 0]
            idx = np.random.choice(initial_idxs)
            obs = self._buffer['observations'][idx]
            return obs, 0

    def reset(self, buffer=None):
        """
        Resets the environments

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        self._buffer = buffer
        if buffer is None:
            results = [self.env.reset() for _ in range(self.num_envs)]  # get initial observation from environment
            self.current_obs = np.stack(results, axis=0)
            assert self.current_obs.ndim == 2
            self.ts[:] = 0
        else:
            initial_idxs = np.arange(len(self._buffer['time_steps']))[self._buffer['time_steps'] == 0]
            idxs = np.random.choice(initial_idxs, size=self.num_envs, replace=True)
            self.current_obs = self._buffer['observations'][idxs]
            self.ts[:] = 0
            results = list(self.current_obs)
        return results

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs
