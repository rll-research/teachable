import numpy as np
from meta_mb.utils.serializable import Serializable
from gym.spaces import Box, Discrete
# from rand_param_envs.gym.spaces import Box as OldBox

"""
Normalizes the environment class.

Args:
    EnvCls (gym.Env): class of the unnormalized gym environment
    env_args (dict or None): arguments of the environment
    scale_reward (float): scale of the reward
    normalize_obs (bool): whether normalize the observations or not
    normalize_reward (bool): whether normalize the reward or not
    obs_alpha (float): step size of the running mean and variance for the observations
    reward_alpha (float): step size of the running mean and variance for the observations

Returns:
    Normalized environment

"""


class RL2Env(Serializable):
    """
    Normalizes the environment class.

    Args:
        Env (gym.Env): class of the unnormalized gym environment
        scale_reward (float): scale of the reward
        normalize_obs (bool): whether normalize the observations or not
        normalize_reward (bool): whether normalize the reward or not
        obs_alpha (float): step size of the running mean and variance for the observations
        reward_alpha (float): step size of the running mean and variance for the observations

    """
    def __init__(self,
                 env,
                 scale_reward=1.,
                 normalize_obs=False,
                 normalize_reward=False,
                 obs_alpha=0.001,
                 reward_alpha=0.001,
                 normalization_scale=10.,
                 ceil_reward=False,
                 ):
        self.ceil_reward = ceil_reward
        Serializable.quick_init(self, locals())

        self._wrapped_env = env
        if isinstance(self._wrapped_env.action_space, Discrete):
            size = self._wrapped_env.action_space.n
        else:
            size = self._wrapped_env.action_space.shape
        self.prev_action = np.zeros(size)
        self.prev_reward = [0]
        self.prev_done = [0]

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        if hasattr(self._wrapped_env, '_wrapped_env'):
            orig_attr = self._wrapped_env.__getattr__(attr)
        else:
            orig_attr = self._wrapped_env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr

    def reset(self):
        obs = self._wrapped_env.reset()
        return np.concatenate([obs, self.prev_action, self.prev_reward, self.prev_done])

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def step(self, action):
        wrapped_step = self._wrapped_env.step(action)
        next_obs, reward, done, info = wrapped_step
        if self.ceil_reward:
            reward = np.ceil(reward) # Send it to 1
        if isinstance(self._wrapped_env.action_space, Discrete):
            ac_idx = action
            action = np.zeros((self._wrapped_env.action_space.n,))
            action[ac_idx] = 1.0
        next_obs_rewardfree = np.concatenate([next_obs, action, [done]]).copy()
        next_obs = np.concatenate([next_obs, action, [reward], [done]]).copy()
        info['next_obs_rewardfree'] = next_obs_rewardfree
        self.prev_action = action
        self.prev_reward = [reward]
        self.prev_done = [done]
        return next_obs, reward, done, info

    def set_task(self, args=None):
        self._wrapped_env.set_task(args=None)
        if isinstance(self._wrapped_env.action_space, Discrete):
            size = self._wrapped_env.action_space.n
        else:
            size = self._wrapped_env.action_space.shape
        self.prev_action = np.zeros(size)
        self.prev_reward = [0]
        self.prev_done = [0]


rl2env = RL2Env