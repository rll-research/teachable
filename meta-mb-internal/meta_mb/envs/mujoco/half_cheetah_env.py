import numpy as np
from meta_mb.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv


class HalfCheetahEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act), axis=1)
            reward_run = obs_next[:, 8]
            reward = reward_run + reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act))
            reward_run = obs_next[8]
            reward = reward_run + reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        else:
            raise NotImplementedError

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.3
        self.viewer.cam.elevation = 3
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.type = 1

    def log_diagnostics(self, paths, prefix=''):
        fwrd_vel = [path["env_infos"]['reward_run'] for path in paths]
        final_fwrd_vel = [path["env_infos"]['reward_run'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgForwardVel', np.mean(fwrd_vel))
        logger.logkv(prefix + 'AvgFinalForwardVel', np.mean(final_fwrd_vel))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))


if __name__ == "__main__":
    env = HalfCheetahEnv()
    env.reset()
    for _ in range(1000):
        img = env.render('rgb_array')
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
