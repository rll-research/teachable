import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import MetaEnv


class ReacherEnv(MetaEnv, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")

        if getattr(self, 'action_space', None):
            a = np.clip(a, self.action_space.low,
                        self.action_space.high)
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        dist_vec = obs[:, -3:]
        reward_dist = - np.linalg.norm(dist_vec, axis=1)
        reward_ctrl = - np.sum(np.square(acts), axis=1)
        reward = reward_dist + reward_ctrl
        return reward

    def tf_reward(self, obs, acts, next_obs):
        dist_vec = obs[:, -3:]
        reward_dist = - tf.linalg.norm(dist_vec, axis=1)
        reward_ctrl = - tf.reduce_sum(np.square(acts), axis=1)
        reward = reward_dist + reward_ctrl
        return reward


if __name__ == "__main__":
    env = ReacherEnv()
    env.reset()
    for _ in range(1000):
        _ = env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
