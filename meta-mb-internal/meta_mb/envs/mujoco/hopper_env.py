import numpy as np
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv

class HopperEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self, 'hopper.xml', 4)
        gym.utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reward(self, obs, act, obs_next):
        alive_bonus = 1.0
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            vel = obs_next[:, 5]
            ctrl_cost = 1e-3 * np.sum(np.square(act), axis=1)
            reward = vel + alive_bonus - ctrl_cost
        else:
            reward = self.reward(np.array([obs]), np.array([act]), np.array([obs_next]))[0]
        return np.minimum(np.maximum(-1000.0, reward), 1000.0)

    def done(self, obs):
        if obs.ndim == 2:
            notdone = np.all(np.isfinite(obs), axis=1) * (np.abs(obs[:, 3:]) < 100).all(axis=1) * (obs[:, 0] > .7) * (np.abs(obs[:, 1]) < .2)
            return np.logical_not(notdone)
        else:
            notdone = np.isfinite(obs).all() and \
                      (np.abs(obs[3:]) < 100).all() and (obs[0] > .7) and \
                      (abs(obs[1]) < .2)
            return not notdone

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.6
        self.viewer.cam.elevation = 3
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.type = 1


if __name__ == "__main__":
    env = HopperEnv()
    while True:
        env.reset()
        for _ in range(200):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
