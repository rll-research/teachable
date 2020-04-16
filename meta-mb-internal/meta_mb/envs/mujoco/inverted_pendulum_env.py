from meta_mb.meta_envs.base import MetaEnv
import os
from gym.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np
from gym import utils

class InvertedPendulumEnv(MetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        # done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        # v = self.viewer
        # v.cam.trackbodyid = 0
        # v.cam.distance = self.model.stat.extent
        pass


if __name__ == "__main__":
    env = InvertedPendulumEnv()
    env.reset()
    for _ in range(1000):
        env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
