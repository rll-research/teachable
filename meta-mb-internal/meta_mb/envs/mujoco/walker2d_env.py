import numpy as np
from meta_mb.meta_envs.base import MetaEnv
from meta_mb.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv

class Walker2DEnv(MetaEnv, gym.utils.EzPickle, MujocoEnv):
    def __init__(self):
        MujocoEnv.__init__(self, 'walker2d.xml', 8)
        gym.utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            vel = obs_next[:, 8]
            alive_bonus = 1.0
            ctrl_cost = 1e-3 * np.sum(np.square(act), axis=1)
            reward = vel - ctrl_cost + alive_bonus

        else:
            reward = self.reward(np.array([obs]), np.array([act]), np.array([obs_next]))[0]
        return reward

    def done(self, obs):
        if obs.ndim == 2:
            notdone = (obs[:, 0] > 0.8) * (obs[:, 0] < 2.0) * (obs[:, 1] > -1.0) * (obs[:, 1] < 1.0)
            return np.logical_not(notdone)
        else:
            return not (obs[0] > 0.8 and obs[0] < 2.0 and obs[1] > -1.0 and obs[1] < 1.0)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.3
        self.viewer.cam.elevation = 3
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.type = 1

if __name__ == "__main__":
    env = Walker2DEnv()
    while True:
        env.reset()
        for _ in range(200):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action