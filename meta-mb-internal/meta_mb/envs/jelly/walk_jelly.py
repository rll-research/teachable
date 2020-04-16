import numpy as np
#from gym.envs.mujoco import mujoco_env
#from gym import utils
import os
import gym
from meta_mb.logger import logger
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv
from meta_mb.meta_envs.base import RandomEnv
from scipy.spatial.transform import Rotation as R

class WalkJellyEnv(RandomEnv, gym.utils.EzPickle):
    def __init__(self, log_rand=0, frameskip=2):
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'jelly.xml')

        #RandomEnv.__init__(self, log_rand,  xml_file, frameskip)
        RandomEnv.__init__(self, 0, xml_file, frameskip)

        self.neutral_lvl = self.get_body_com("base_link")[2]
        self.upper_limit = self.neutral_lvl + int(self.neutral_lvl/2)
        self.lower_limit = self.neutral_lvl - int(self.neutral_lvl/2)

        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.body_position()
        ])

    def step(self, action): 
        self.prev_pos = self.body_position()[0]
        self.do_simulation(action, self.frame_skip)
        self.curr_pos = self.body_position()[0]

        # reward_stand = -self.standing() #first attempt at reward function

        quat = self.sim.data.get_body_xquat("base_link")
        r = R.from_quat(quat=quat)
        penalty_roll = r.as_rotvec()[0]

        penalty_acc = np.sum(self.sim.data.qacc[6:])

        reward_dist = self.curr_pos - self.prev_pos
        reward_ctrl = -0.5*np.square(action).sum()
        reward = reward_dist - (0.5 * penalty_roll) - (0.05 * penalty_acc) - reward_ctrl

        state = self.state_vector()
        notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.0

        observation = self._get_obs()
        done = not notdone
        info = dict(reward_dist=reward_dist,
                    reward_ctrl=reward_ctrl,
                    penalty_roll=penalty_roll,
                    penalty_acc=penalty_acc)

        return observation, reward, done, info

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            notdone = np.isfinite(obs_next).all() and (np.abs(obs_next[:, 1]) <= .2)
            reward = notdone.astype(np.float32)
            return reward

        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]

        else:
            raise NotImplementedError

    def standing(self):
        reward = 0

        neutral_lvl = self.get_body_com("base_link")[2]
        upper_limit = neutral_lvl + int(neutral_lvl/6)
        lower_limit = neutral_lvl - 2*int(neutral_lvl/6)

        hip_heights = self.get_hip_heights()
        for hip in hip_heights:
            if hip < lower_limit or hip > upper_limit:
                reward -= 1
        return reward

    def get_hip_heights(self):
        return np.array([
            self.get_body_com("FR_hip_link")[2],
            self.get_body_com("FL_hip_link")[2],
            self.get_body_com("RR_hip_link")[2],
            self.get_body_com("RL_hip_link")[2]
            ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.sim.model.body_pos[-1] = np.append(np.random.uniform(-5, 5, 2), np.random.uniform(0, 0.15))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def body_position(self):
        return self.get_body_com("base_link")

if __name__ == "__main__":
    env = WalkJellyEnv()
    while True:
        env.reset()
        for _ in range(800):
            action = env.action_space.sample()
            env.step(action)
            env.render()