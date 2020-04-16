import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os
from scipy.spatial.distance import euclidean
from meta_mb.meta_envs.base import RandomEnv
#from mujoco-py.mujoco_py.pxd.mujoco import local
import mujoco_py

class PegFullBlueEnv(RandomEnv, utils.EzPickle):
    def __init__(self, goal_dist=3e-2):
        utils.EzPickle.__init__(**locals())

        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_full_peg_v1.xml')

        x = 0.005
        y = -0.5
        z = -0.35

        self.top_goal = np.array([x, y, z+0.15])
        self.center_goal = np.array([x, y, z])
        self.bottom_goal= np.array([x, y, z-0.15])

        self.peg_loc = self.center_goal

        self.goal_dist = goal_dist # permissible distance from goal

        RandomEnv.__init__(self, 2, xml_file, 2)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat[:-3],
            #self.sim.data.body_xpos.flat[:3],
            self.peg_location() - self.center_goal
        ])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        self.peg_loc = self.peg_location()
        reward_dist = -self.peg_dist()
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 1.25e-4 * reward_ctrl
        self.peg_orient()

        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        print(reward)
        return observation, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)

        peg_table_position = np.random.uniform(low=[-0.2, -1, 0.3], high=[0.75, -0.6, 0.3])
        self.sim.model.body_pos[-8] = peg_table_position

        self.top_goal = self.get_body_com("g1")
        self.center_goal = self.get_body_com("g2")
        self.bottom_goal = self.get_body_com("g3")

        qpos[-6:-3] = np.zeros((3, ))
        qpos[-3:] = self.center_goal
        qvel[-6:] = 0

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_ctrl = -np.sum(np.square(act), axis=1)
            reward_dist = -self.peg_dist()
            reward = reward_dist + 1.25e-4 * reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            reward_ctrl = -np.sum(np.square(act))
            reward_dist = -self.peg_dist()
            reward = reward_run + 1.25e-4 * reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        else:
            raise NotImplementedError

    def peg_orient(self):
        return self.data.get_body_xquat("peg-center")

    def peg_dist(self):
        top = self.get_body_com("peg-top")
        center = self.get_body_com("peg-center")
        bottom = self.get_body_com("peg-bottom")

        distance = (euclidean(top, self.top_goal)
                    + euclidean(center, self.center_goal)
                    + euclidean(bottom, self.bottom_goal))
        return distance

    def peg_location(self):
        return self.get_body_com("peg-center")

    def top(self, center):
        x = center[0]
        y = center[1] + 0.3
        z = center[2] - 0.4
        return np.array([x, y, z])

    def center(self, center):
        x = center[0]
        y = center[1] + 0.3
        z = center[2] - 0.55
        return np.array([x, y, z])

    def bottom(self, center):
        x = center[0]
        y = center[1] + 0.3
        z = center[2] - 0.7
        return np.array([x, y, z])

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0
        self.viewer.cam.azimuth = 180

if __name__ == "__main__":
    env = PegFullBlueEnv()
    while True:
        env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()