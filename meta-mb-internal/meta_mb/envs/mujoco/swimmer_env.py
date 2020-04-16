import numpy as np
import gym
from meta_mb.logger import logger
from meta_mb.meta_envs.base import MetaEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv

class SwimmerEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self, 'swimmer.xml', 4)
        gym.utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = np.abs((xposafter - xposbefore) / self.dt)
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def reward(self, obs, act, obs_next):
        ctrl_cost_coeff = 0.0001
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            vel = obs_next[:, 3]
            ctrl_cost = ctrl_cost_coeff * np.sum(np.square(act), axis=1)
            reward = vel - ctrl_cost
        else:
            reward = self.reward(np.array([obs]), np.array([act]), np.array([obs_next]))[0]
        return np.minimum(np.maximum(-1000.0, reward), 1000.0)
        
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

