import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from meta_mb.meta_envs.base import RandomEnv
import os
import time


class FullBlueEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_rand=0, timeskip=20, actions=None):
        utils.EzPickle.__init__(**locals())

        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_full_v1.xml')
        max_torques = np.array([10, 10, 8, 6, 6, 4, 4])
        self.goal_right = np.zeros(3)
        self._low = -max_torques
        self._high = max_torques

        if actions is not None:
            self.actions = actions
            self.path_len = 0
            self.max_path_len = len(actions)

        RandomEnv.__init__(self, log_rand, xml_file, timeskip)
        #RandomEnv.__init__(self, 0, xml_file, 2)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat[:-3],
            self.get_body_com("right_gripper_link"),
            self.ee_position('right') - self.goal_right,
        ])

    def step(self, action):
        done = False

        if (hasattr(self, "actions")):
            action = self.actions[self.path_len]
            self.path_len += 1
            if(self.path_len == self.max_path_len):
                done = True

        self.do_simulation(action, self.frame_skip)
        # self.correction() #Use for v2 arms
        vec_right = self.ee_position('right') - self.goal_right
        reward_dist = -np.linalg.norm(vec_right)
        reward_ctrl = -np.square(action/(2* self._high)).sum()
        reward = reward_dist + 0.5 * 0.1 * reward_ctrl
        observation = self._get_obs()

        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reset_model(self):
        self.frame_skip = np.random.randint(1, 5) #randomize frameskips
        gravity = np.random.randint(-4, 1) #randomize environment gravity
        self.model.opt.gravity[2] = gravity
        self.sim.data.qfrc_applied[-1] = abs(gravity/1.90986) #counteract gravity on goal body

        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.goal_right = np.random.uniform(low=[0.25, -0.75, 0.25], high=[0.75, -0.25, 0.5])
        #self.goal_right = np.array([.65, -0.5, .41]) #fixed goal
        qpos[-6:-3] = 0
        qpos[-3:] = self.goal_right
        qvel[-6:] = 0

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def ee_position(self, arm):
        return (self.get_body_com(arm + '_r_finger_tip_link')
                + self.get_body_com(arm + '_l_finger_tip_link'))/2

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act/(2 * self._high)), axis=1)
            reward_dist = -np.linalg.norm(obs_next[:, -3:], axis=1)
            reward = reward_dist + reward_ctrl
            return np.clip(reward, -1e2, 1e2)

        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]
        
        else:
            raise NotImplementedError

    def correction(self):
        try:
            qpos = self.sim.data.qpos.flat
            qvel = self.sim.data.qvel.flat[:-3]
            shoulder_lift_joint = qpos[1]
            if shoulder_lift_joint > -0.01:
                correction = 0.5 * abs(shoulder_lift_joint) + shoulder_lift_joint
            qpos[1] = correction
            qvel[1] = qvel[1] * 0.9
            self.set_state(qpos, qvel)
        except:
            pass


    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0
        self.viewer.cam.azimuth = 180


if __name__ == "__main__":
    env = FullBlueEnv()
    while True:
        env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            env.step(action)
            env.render()
