from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import MetaEnv


class FWalker2dEnv(MetaEnv, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, frame_skip=4):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/walker2d.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = old_ob[8]
        reward_height = -3.0 * np.square(old_ob[0] - 1.3)

        height, ang = ob[0], ob[1]
        done = (height >= 2.0) or (height <= 0.8) or (abs(ang) >= 1.0)
        alive_reward = float(not done)

        reward = reward_run + reward_ctrl + reward_height + alive_reward

        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == acts.ndim == next_obs.ndim < 3
        assert obs.shape == next_obs.shape

        if obs.ndim == 2:
            reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
            reward_run = obs[:, 8]
            reward_height = -3.0 * np.square(next_obs[:, 0] - 1.3)

            height, ang = next_obs[:, 0], next_obs[:, 1]
            done = np.logical_or(
                np.logical_or(height >= 2.0, height <= 0.8),
                np.abs(ang) >= 1.0
            )
            alive_reward = 1.0 - np.array(done, dtype=np.float)

            reward = reward_run + reward_ctrl + reward_height + alive_reward
            return reward
        elif obs.ndim == 1:
            obs = np.expand_dims(obs, 0)
            acts = np.expand_dims(acts, 0)
            next_obs = np.expand_dims(next_obs, 0)
            return self.reward(obs, acts, next_obs)[0]
        
    def tf_reward(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * tf.reduce_sum(np.square(acts), axis=1)
        reward_run = obs[:, 8]
        reward_height = -3.0 * tf.square(next_obs[:, 0] - 1.3)

        height, ang = next_obs[:, 0], next_obs[:, 1]
        done = tf.math.logical_or(
            tf.math.logical_or(tf.math.greater_equal(height, 2.0), tf.math.greater_equal(0.8, height)),
            tf.math.greater_equal(tf.abs(ang), 1.0)
        )
        alive_reward = 1.0 - tf.cast(done, dtype=tf.float32)
        reward = reward_run + reward_ctrl + reward_height + alive_reward
        return reward


if __name__ == "__main__":
    env = FWalker2dEnv()
    env.reset()
    for _ in range(1000):
        _ = env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
