# Allow us to interact wth the PointMassEnv the same way we interact with the TeachableRobotLevels class.
import numpy as np
from copy import deepcopy
import gym
import d4rl

class PointMassEnv:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, env_name, feedback_type=None, feedback_freq=False, intermediate_reward=False,
                 cartesian_steps=[1], **kwargs):
        self._wrapped_env = gym.make(env_name)
        self.feedback_type = feedback_type
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))  # TODO: seed isn't passed in
        # TODO: create teachers

    def step(self, action):
        obs, rew, done, info = self._wrapped_env.step(action)
        obs_dict = {}
        obs_dict["obs"] = obs
        # info['teacher_action'] = np.array([self.action_space.n], dtype=np.int32)
        # info['gave_reward'] = int(provided_reward)
        self.done = done
        info['success'] = False  # TODO: eventually compute this
        info['gave_reward'] = True
        return obs_dict, rew, done, info

    def set_task(self, *args, **kwargs):
        pass  # for compatibility with babyai, which does set tasks

    def reset(self):
        obs = self._wrapped_env.reset()
        obs_dict = {'obs': obs}
        return obs_dict

    def render(self, mode='human'):
        try:
            self._wrapped_env.render(mode)
        except Exception as e:  # Probably offscreen
            print(f"Error trying to render {e}")

    def vocab(self):  # We don't have vocab
        return [0]

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        # """
        try:
            if attr == '__len__':
                return None
            results = self.__getattribute__(attr)
            return results
        except:
            orig_attr = self._wrapped_env.__getattribute__(attr)

            if callable(orig_attr):
                def hooked(*args, **kwargs):
                    result = orig_attr(*args, **kwargs)
                    return result

                return hooked
            else:
                return orig_attr
