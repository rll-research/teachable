import numpy as np
from gym.spaces import Box, Discrete


class DummyDiscrete:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, **kwargs):
        self.timesteps = 0
        self.time_limit = 1
        self.teacher_action = np.array(-1)
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([1, 1]))
        self.action_space = Discrete(2)
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))
        self.index = 0
        self.successes = 0

    def seed(self, *args, **kwargs):
        pass

    def get_timestep(self):
        return .02

    def step(self, action):
        if action == 1:
            rew = 1
            self.successes += 1
        else:
            rew = 0
        self.timesteps += 1
        done = self.timesteps >= self.time_limit
        self.index = np.random.randint(2)
        # print("index", self.index)
        state = np.zeros(3, dtype=np.float32)
        state[self.index] = 1
        obs_dict = {'obs': state,
                    'Direction': np.array([0.0])}
        success = done and self.successes == self.timesteps
        if done:
            # print("num succ", self.successes)
            self.successes = 0
        info = {}
        info['success'] = success
        info['gave_reward'] = True
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self.timesteps
        info['next_obs'] = obs_dict
        return obs_dict, rew, done, info

    def reset(self):
        self.index = np.random.randint(2)
        # print("index", self.index)
        state = np.zeros(3, dtype=np.float32)
        state[self.index] = 1
        self.timesteps = 0
        obs_dict = {'obs': state,
                    'Direction': np.array([0.0])}
        return obs_dict

    def render(self, mode='human'):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[48:52, 48:52, :2] = 1
        y = int(min(98, max(2, np.round(self.index * 10) + 50)))
        x = y
        # x = int(min(98, max(2, np.round(self.pos[1] * 10) + 50)))
        img[y - 2: y + 2, x - 2: x + 2] = 1
        return img * 255

    def vocab(self):  # We don't have vocab
        return [0]


class PointMassEnvSimple:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, **kwargs):
        self.timesteps = 0
        self.time_limit = 10
        self.target = np.array([0], dtype=np.float32)
        self.pos = np.array([3], dtype=np.float32)
        self.teacher_action = np.array(-1)
        self.observation_space = Box(low=np.array([-5, -5]), high=np.array([5, 5]))
        self.action_space = Box(low=np.array([-1]), high=np.array([1]))
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))

    def seed(self, *args, **kwargs):
        pass

    def get_timestep(self):
        return .02

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.pos += action
        rew = -np.linalg.norm(self.target - self.pos) / 10
        self.timesteps += 1
        done = self.timesteps >= self.time_limit
        obs = self.pos.copy()
        obs_dict = {'obs': obs,
                    'Direction': np.array([0.0])}
        reached_goal = np.linalg.norm(self.target - self.pos) < .49
        success = done and reached_goal
        info = {}
        info['success'] = success
        info['gave_reward'] = True
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self.timesteps
        info['next_obs'] = obs_dict
        return obs_dict, rew, done, info

    def reset(self):
        self.pos = np.array([3], dtype=np.float32)
        self.timesteps = 0
        obs_dict = {'obs': self.pos.copy(),
                    'Direction': np.array([0.0])}
        return obs_dict

    def render(self, mode='human'):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[48:52, 48:52, :2] = 1
        y = int(min(98, max(2, np.round(self.pos[0] * 10) + 50)))
        x = y
        # x = int(min(98, max(2, np.round(self.pos[1] * 10) + 50)))
        img[y - 2: y + 2, x - 2: x + 2] = 1
        return img * 255

    def vocab(self):  # We don't have vocab
        return [0]