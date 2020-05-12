from meta_mb.utils.serializable import Serializable
from babyai.levels.iclr19_levels import *
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


class Curriculum(Serializable):
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
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())
        self.levels_list = [Level_GoToRedBallGrey(**kwargs),
                            Level_GoToRedBallNoDists(**kwargs),
                            Level_GoToRedBall(**kwargs),
                            Level_GoToObjS4(**kwargs),
                            Level_GoToObjS6(**kwargs),
                            Level_GoToObj(**kwargs),
                            Level_GoToLocalS5N2(**kwargs),
                            Level_GoToLocalS6N2(**kwargs),
                            Level_GoToLocalS6N3(**kwargs),
                            Level_GoToLocalS6N4(**kwargs),
                            Level_GoToLocalS7N4(**kwargs),
                            Level_GoToLocalS7N5(**kwargs),
                            Level_GoToLocalS8N2(**kwargs),
                            Level_GoToLocalS8N3(**kwargs),
                            Level_GoToLocalS8N4(**kwargs),
                            Level_GoToLocalS8N5(**kwargs),
                            Level_GoToLocalS8N6(**kwargs),
                            Level_GoToLocalS8N7(**kwargs),
                            Level_GoToLocal(**kwargs),
                            Level_PutNextLocal(**kwargs),
                            Level_PutNextLocalS5N3(**kwargs),
                            Level_PutNextLocalS6N4(**kwargs),
                            Level_PutNextLocal(**kwargs),
                            Level_Open(**kwargs),
                            Level_GoToOpen(**kwargs),
                            Level_GoToObjMaze(**kwargs),
                            Level_GoToObjMazeOpen(**kwargs),
                            Level_GoToObjMazeS4R2(**kwargs),
                            Level_GoToObjMazeS4(**kwargs),
                            Level_GoToObjMazeS5(**kwargs),
                            Level_GoToObjMazeS6(**kwargs),
                            Level_GoToObjMazeS6(**kwargs),
                            Level_GoTo(**kwargs),
                            Level_Unlock(**kwargs),
                            Level_GoToImpUnlock(**kwargs),
                            Level_Pickup(**kwargs),
                            Level_PutNext(**kwargs),
                            Level_UnblockPickup(**kwargs),
                            ]
        self._wrapped_env = self.levels_list[0]

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
        return self._wrapped_env.reset()

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def step(self, action):
        return self._wrapped_env.step(action)

    def advance_curriculum(self):
        """Currently we just advance one-by-one when this function is called.
        Later, it would be cool to advance dynamically when the agent has succeeded at a task.
        """
        self._wrapped_env = self.levels_list[self.index]
        self.index += 1
        print("updated curriculum")
