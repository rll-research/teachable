import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os
from scipy.spatial.distance import euclidean
from meta_mb.meta_envs.base import RandomEnv
# from mujoco-py.mujoco_py.pxd.mujoco import local
import mujoco_py


class PegArmBlueEnv(RandomEnv, utils.EzPickle):
    def __init__(self, goal_dist=0.1, log_rand=0, frame_skip=25):
        utils.EzPickle.__init__(**locals())

        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_right_peg_v2.xml')

        self.peg_loc = np.zeros(3)
        self.reach_goal = np.zeros(3)
        self.peg_board = np.zeros(3)
        self.goal_dist = goal_dist  # permissible distance from goal

        self.reached = False

        RandomEnv.__init__(self, log_rand, xml_file, frame_skip)


    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("peg"),
            self.peg_location() - self.peg_board
        ])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.peg_loc = self.peg_location()

        if not self.reached:
            reward_dist = -1.5 * self.reach_dist()
        else:
            reward_dist = -self.peg_insertion()

        joint_velocities = self.sim.data.qvel
        reward_ctrl = -np.square(joint_velocities).sum()
        reward = reward_dist + 1.25e-3 * reward_ctrl

        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)

        self.peg_board = self.random_pos()
        self.sim.model.body_pos[-4] = self.peg_board
        self.sim.model.body_pos[-3] = self.g1()
        self.sim.model.body_pos[-2] = self.g2()
        self.sim.model.body_pos[-1] = self.g3()

        observation = self._get_obs()
        return observation

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            joint_velocities = self.sim.data.qvel
            reward_ctrl = -np.sum(np.square(joint_velocities), axis=1)
            if not self.reached:
                reward_dist = -1.5 * self.reach_dist()
            else:
                reward_dist = -self.peg_insertion()
            reward = reward_dist + 1.25e-3 * reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            reward_ctrl = -np.sum(np.square(act))
            reward_dist = -self.peg_dist()
            reward = reward_dist + 1.25e-4 * reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        else:
            raise NotImplementedError

    def random_pos(self):
        x = np.random.uniform(low=-0.3, high=0.3)
        y = np.random.uniform(low=-0.25, high=0.25)
        if abs(x) < 0.1:
            sign = x / abs(x)
            x += 0.2 * sign
        if abs(y) < 0.1:
            sign = y / abs(y)
            y += 0.2 * sign
        return np.array([x, y, 0.01])

    def peg_location(self):
        return self.get_body_com("peg")

    def peg_orient(self):
        return self.data.get_body_xquat("peg")

    def reach_dist(self):
        center = self.get_body_com("center")
        g1 = self.get_body_com("g1")

        dist = euclidean(center, g1)
        if dist < self.goal_dist:
            self.reached = True
        return dist

    def peg_insertion(self):
        top = self.get_body_com("top")
        bottom = self.get_body_com("bottom")
        g2 = self.get_body_com("g2")
        g3 = self.get_body_com("g3")

        return euclidean(top, g2) + euclidean(bottom, g3)

    def g1(self):
        x = 0.092
        y = 0
        z = 0.75
        return np.array([x, y, z])

    def g2(self):
        x = 0.092
        y = 0 
        z = 0.078
        return np.array([x, y, z])

    def g3(self):
        x = 0.092
        y = 0
        z = 0
        return np.array([x, y, z])

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0
        self.viewer.cam.azimuth = 180


if __name__ == "__main__":
    env = PegArmBlueEnv()
    while True:
        env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()