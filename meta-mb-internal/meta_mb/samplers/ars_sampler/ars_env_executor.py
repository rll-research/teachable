import numpy as np
import copy


class ModelARSEnvExecutor(object):
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

    def __init__(self, env, dynamics_model, num_deltas, rollouts_per_policy, max_path_length):
        self.env = env
        self.dynamics_model = dynamics_model
        self.rollouts_per_policy = rollouts_per_policy
        self.num_deltas = num_deltas
        self._num_envs = 2 * num_deltas * rollouts_per_policy
        self.action_dim = np.prod(env.action_space.shape)
        self.obs_dim = np.prod(env.observation_space.shape)

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        # check whether env has done function
        self.has_done_fn = hasattr(self.unwrapped_env, 'done')

        self.ts = np.zeros(2 * num_deltas * rollouts_per_policy, dtype='int')  # time steps
        self.max_path_length = max_path_length
        self.current_obs = None
        self._buffer = None

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions

        Args:
            actions (np.array): array of actions with shape (num_deltas, batch_size, action_dim)

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of meta_envs)
        """
        assert actions.shape == (2 * self.num_deltas, self.rollouts_per_policy, self.action_dim)

        actions = np.reshape(actions.transpose((1, 0, 2)), (-1, self.action_dim))

        prev_obs = self.current_obs
        next_obs = self.dynamics_model.predict_batches(prev_obs, actions)
        rewards = self.unwrapped_env.reward(prev_obs, actions, next_obs)

        if self.has_done_fn:
            dones = self.unwrapped_env.done(next_obs)
        else:
            dones = np.asarray([False for _ in range(self.num_envs)])

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)
        for i in np.argwhere(dones).flatten():
            next_obs[i] = self._reset(i)
            self.ts[i] = 0

        self.current_obs = next_obs

        next_obs = np.reshape(next_obs, (self.rollouts_per_policy, 2 * self.num_deltas, self.obs_dim)).transpose((1, 0, 2))
        rewards = np.reshape(rewards, (self.rollouts_per_policy, 2 * self.num_deltas)).T

        return next_obs, rewards, dones, {}

    def set_tasks(self, tasks):
        """
        Sets a list of tasks to each environment

        Args:
            tasks (list): list of the tasks for each environment
        """
        pass

    def _reset(self, i):
        if self._buffer is None:
            return self.env.reset()

        else:
            idx = np.random.randint(len(self._buffer['observations']))
            return self._buffer['observations'][idx]

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
        else:
            idx = np.random.randint(0, len(self._buffer['observations']), size=self.rollouts_per_policy)
            self.current_obs = np.concatenate([self._buffer['observations'][idx]] * self.num_deltas * 2)
        assert self.current_obs.ndim == 2
        self.ts[:] = 0
        results = np.reshape(self.current_obs, (self.rollouts_per_policy, 2 * self.num_deltas, self.obs_dim)).transpose(
            (1, 0, 2))
        return results

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs


class IterativeARSEnvExecutor(object):
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

    def __init__(self, env, num_deltas, rollouts_per_policy, max_path_length):
        self.env = env
        self.rollouts_per_policy = rollouts_per_policy
        self.num_deltas = num_deltas
        self._num_envs = 2 * num_deltas * rollouts_per_policy

        self.envs = np.asarray([copy.deepcopy(env) for _ in range(self._num_envs)])

        self.action_dim = np.prod(env.action_space.shape)
        self.obs_dim = np.prod(env.observation_space.shape)

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        # check whether env has done function
        self.has_done_fn = hasattr(self.unwrapped_env, 'done')

        self.ts = np.zeros(2 * num_deltas * rollouts_per_policy, dtype='int')  # time steps
        self.max_path_length = max_path_length
        self.current_obs = None
        self._buffer = None

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions

        Args:
            actions (np.array): array of actions with shape (num_deltas, batch_size, action_dim)

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of meta_envs)
        """
        assert actions.shape == (2 * self.num_deltas, self.rollouts_per_policy, self.action_dim)

        actions = np.reshape(actions.transpose((1, 0, 2)), (-1, self.action_dim))

        prev_obs = self.current_obs
        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)]

        # stack results split to obs, rewards, ...
        next_obs, rewards, dones, env_infos = list(map(list, zip(*all_results)))

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)
        for i in np.argwhere(dones).flatten():
            next_obs[i] = self._reset(i)
            self.ts[i] = 0

        self.current_obs = next_obs

        next_obs = np.reshape(next_obs, (self.rollouts_per_policy, 2 * self.num_deltas, self.obs_dim)).transpose((1, 0, 2))
        rewards = np.reshape(rewards, (self.rollouts_per_policy, 2 * self.num_deltas)).T

        return next_obs, rewards, dones, {}

    def set_tasks(self, tasks):
        """
        Sets a list of tasks to each environment

        Args:
            tasks (list): list of the tasks for each environment
        """
        pass

    def _reset(self, i):
        if self._buffer is None:
            return self.envs[i].reset()

        else:
            idx = np.random.randint(len(self._buffer['observations']))
            return self.envs[i].reset_from_obs(self._buffer['observations'][idx])

    def reset(self, buffer=None):
        """
        Resets the environments

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        self._buffer = buffer
        results = [env.reset() for env in self.envs]  # get initial observation from environment
        if buffer is None:
            self.current_obs = np.stack(results, axis=0)
        else:
            idx = np.random.randint(0, len(self._buffer['observations']), size=self.rollouts_per_policy)
            # init_mask = np.random.binomial(1, 1/self.rollouts_per_policy, size=(self.rollouts_per_policy, 1))
            # if sum(init_mask) > 0:
            #     obses_per_rollout = self._buffer['observations'][idx] * (1 - init_mask) + init_mask * np.array([self.env.reset()
            #                                                                                                 for _ in range(int(sum(init_mask)))])
            # else:
            obses_per_rollout = self._buffer['observations'][idx]
            obses = np.concatenate([obses_per_rollout] * self.num_deltas * 2)
            self.current_obs = np.array([env.reset_from_obs(obs) for obs, env in zip(obses, self.envs)])
        assert self.current_obs.ndim == 2
        self.ts[:] = 0
        results = np.reshape(self.current_obs, (self.rollouts_per_policy, 2 * self.num_deltas, self.obs_dim)).transpose(
            (1, 0, 2))
        return results

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs
