# Allow us to interact wth the D4RLEnv the same way we interact with the TeachableRobotLevels class.
import numpy as np
from copy import deepcopy
import gym
import d4rl
from gym.spaces import Box, Discrete


class PointMassEnvSimple:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, env_name, feedback_type=None, feedback_freq=False, intermediate_reward=False,
                 cartesian_steps=[1], **kwargs):
        self.timesteps = 0
        self.time_limit = 10
        self.target = np.array([0, 0], dtype=np.float32)
        self.pos = np.array([3, 4], dtype=np.float32)
        self.feedback_type = feedback_type
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))  # TODO: seed isn't passed in
        self.teacher_action = np.array(-1)
        self.observation_space = Box(low=np.array([-5, -5]), high=np.array([5, 5]))
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]))
        # TODO: create teachers

    def seed(self, *args, **kwargs):
        pass

    def step(self, action):
        action = np.clip(action, -1, 1)
        if action.shape == (1, 2):
            action = action[0]
        self.pos += action
        rew = -np.linalg.norm(self.target - self.pos) / 10
        self.timesteps += 1
        done = self.timesteps >= self.time_limit
        obs = self.pos
        obs_dict = {'obs': obs}
        success = done and np.linalg.norm(self.target - self.pos) < .49
        info = {}
        info['success'] = success
        info['gave_reward'] = True
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self.timesteps
        return obs_dict, rew, done, info

    def set_task(self, *args, **kwargs):
        pass  # for compatibility with babyai, which does set tasks

    def reset(self):
        self.pos = np.array([3, 4], dtype=np.float32)
        self.timesteps = 0
        obs_dict = {'obs': self.pos}
        return obs_dict

    def render(self, mode='human'):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[48:52, 48:52, :2] = 1
        y = int(min(98, max(2, np.round(self.pos[0] * 10) + 50)))
        x = int(min(98, max(2, np.round(self.pos[1] * 10) + 50)))
        img[y - 2: y + 2, x - 2: x + 2] = 1
        return img * 255

    def vocab(self):  # We don't have vocab
        return [0]


class PointMassEnvSimpleDiscrete:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, env_name, feedback_type=None, feedback_freq=False, intermediate_reward=False,
                 cartesian_steps=[1], **kwargs):
        self.timesteps = 0
        self.time_limit = 10
        self.target = np.array([0, 0], dtype=np.float32)
        self.pos = np.array([3, 4], dtype=np.float32)
        self.feedback_type = feedback_type
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))  # TODO: seed isn't passed in
        self.teacher_action = np.array(-1)
        self.observation_space = Box(low=np.array([-5, -5]), high=np.array([5, 5]))
        self.action_space = Discrete(5)
        # TODO: create teachers

    def seed(self, *args, **kwargs):
        pass

    def step(self, action):
        if action == 0:
            action = np.array([-1, 0])
        elif action == 1:
            action = np.array([1, 0])
        elif action == 2:
            action = np.array([0, -1])
        elif action == 3:
            action = np.array([0, 1])
        elif action == 4:
            action = np.array([0, 0])
        else:
            print("uh oh")
        self.pos += action
        rew = -np.linalg.norm(self.target - self.pos) / 10
        self.timesteps += 1
        done = self.timesteps >= self.time_limit
        obs = self.pos
        obs_dict = {'obs': obs}
        success = done and np.linalg.norm(self.target - self.pos) < .49
        info = {}
        info['success'] = success
        info['gave_reward'] = True
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self.timesteps
        return obs_dict, rew, done, info

    def set_task(self, *args, **kwargs):
        pass  # for compatibility with babyai, which does set tasks

    def reset(self):
        self.pos = np.array([3, 4], dtype=np.float32)
        self.timesteps = 0
        obs_dict = {'obs': self.pos}
        return obs_dict

    def render(self, mode='human'):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[48:52, 48:52, :2] = 1
        y = int(min(98, max(2, np.round(self.pos[0] * 10) + 50)))
        x = int(min(98, max(2, np.round(self.pos[1] * 10) + 50)))
        img[y - 2: y + 2, x - 2: x + 2] = 1
        return img * 255

    def vocab(self):  # We don't have vocab
        return [0]


class D4RLEnv:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, env_name, feedback_type=None, feedback_freq=False, intermediate_reward=False,
                 cartesian_steps=[1], **kwargs):
        self._wrapped_env = gym.envs.make(env_name, reset_target=True)
        self.feedback_type = feedback_type
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))  # TODO: seed isn't passed in
        self.teacher_action = np.array(-1)
        # TODO: create teachers

    def get_target(self):
        raise NotImplementedError

    def step(self, action):
        obs, rew, done, info = self._wrapped_env.step(action)
        obs_dict = {}
        obs_dict["obs"] = obs
        self.done = done

        target = self.get_target()
        agent_pos = obs[:2]
        success = done and np.linalg.norm(target - agent_pos) < .5
        info = {}
        info['success'] = success
        info['gave_reward'] = True
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self._wrapped_env._elapsed_steps
        return obs_dict, rew, done, info

    def set_task(self, *args, **kwargs):
        pass  # for compatibility with babyai, which does set tasks

    def reset(self):
        obs = self._wrapped_env.reset()
        obs_dict = {'obs': obs}
        return obs_dict
        # return obs

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


class PointMassEnv(D4RLEnv):
    def __init__(self, *args, **kwargs):
        super(PointMassEnv, self).__init__(*args, **kwargs)
        # Adding goal
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(6,))

    def get_target(self):
        return self._wrapped_env.get_target()

    def step(self, action):
        obs_dict, rew, done, info = super().step(action)
        obs_dict['obs'] = np.concatenate([obs_dict['obs'], self.get_target()])
        rew = rew / 10 - .01
        return obs_dict, rew, done, info

    def reset(self):
        obs_dict = super().reset()
        obs_dict['obs'] = np.concatenate([obs_dict['obs'], self.get_target()])
        return obs_dict


class AntEnv(D4RLEnv):
    def __init__(self, env_name, sparse=True, **kwargs):
        super().__init__(env_name, **kwargs)
        reward_type = 'sparse' if sparse else 'dense'
        self.reward_type = reward_type
        self._wrapped_env = gym.envs.make(env_name, reset_target=True, reward_type=reward_type)

    def get_target(self):
        return self._wrapped_env.target_goal

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def step(self, action):
        obs_dict, rew, done, info = super().step(action)
        if self.reward_type == 'dense':
            rew = rew / 100 + .1
        return obs_dict, rew, done, info
