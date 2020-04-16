from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import MetaEnv


class FSwimmerEnv(MetaEnv, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, frame_skip=4):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/fixed_swimmer.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        ctrl_cost_coeff = 0.0001

        """
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        """

        self.xposbefore = self.sim.data.site_xpos[0][0] / self.dt
        self.do_simulation(action, self.frame_skip)
        self.xposafter = self.sim.data.site_xpos[0][0] / self.dt
        self.pos_diff = self.xposafter - self.xposbefore

        reward_fwd = self.xposafter - self.xposbefore
        reward_ctrl = - ctrl_cost_coeff * np.square(action).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd,
                                       reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat, self.pos_diff.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        reward_ctrl = -0.0001 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, -1]
        reward = reward_run + reward_ctrl
        return reward

    def tf_reward(self, obs, acts, next_obs):
        reward_ctrl = -0.0001 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = next_obs[:, -1]
        reward = reward_run + reward_ctrl
        return reward


if __name__ == "__main__":
    env = FSwimmerEnv()
    env.reset()
    for _ in range(1000):
        _ = env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
