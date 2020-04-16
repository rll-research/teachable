import numpy as np
#from gym.envs.mujoco import mujoco_env
#from gym import utils
import os
import gym
from meta_mb.logger import logger
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv
from meta_mb.meta_envs.base import RandomEnv

class FetchJellyEnv(RandomEnv, gym.utils.EzPickle):
    def __init__(self):
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'jelly.xml')

        self.goal = np.append(np.random.uniform(-5, 5, 2), np.random.uniform(0, 0.15))
        RandomEnv.__init__(self, 0,  xml_file, 2)
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.sim.data.body_xpos.flat[:3],
            self.get_body_com("base_link") - self.goal
        ])

    def step(self, action):
        self.prev_pos = self.get_body_com("base_link")
        self.do_simulation(action, self.frame_skip)
        self.curr_pos = self.get_body_com("base_link")

        vec_to_goal = self.get_body_com("base_link") - self.goal

        self.reward_dist = -(np.linalg.norm(self.curr_pos - self.prev_pos)) / self.dt
        reward_ctrl = -0.5*0.1*np.square(action).sum()
        reward_fetch = -np.linalg.norm(vec_to_goal)
        reward = 1.25e-4 * (reward_dist - reward_ctrl + 1) + reward_fetch

        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            vec_to_goal = self.get_body_com("base_link") - self.goal
            reward_dist = -(np.linalg.norm(self.curr_pos - self.prev_pos)) / self.dt
            reward_ctrl = -0.5*0.1*np.square(action).sum()
            reward_fetch = -np.linalg.norm(vec_to_goal)
            reward = 1.25e-4 * (forward_progress - cost + 1) + reward_fetch
            return reward

        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]

        else:
            raise NotImplementedError

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.sim.model.body_pos[-1] = np.append(np.random.uniform(-5, 5, 2), np.random.uniform(0, 0.15))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def body_position(self):
        return self.get_body_com("base_link")

if __name__ == "__main__":
    env = FetchJellyEnv()
    while True:
        env.reset()
        for _ in range(10000):
            action = env.action_space.sample()
            env.step(action)
            env.render()


