import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import tensorflow as tf
from os import path
from meta_mb.meta_envs.base import MetaEnv


class PendulumO01Env(MetaEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta
        '''
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
        '''

        # for the reward
        y, x, thetadot = np.cos(th), np.sin(th), thdot
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = y + .1 * np.abs(x) + .1 * (thetadot ** 2) + .001 * (u ** 2)
        reward = -costs

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        self.last_u = u  # for rendering
        # costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        ob = np.array(self._get_obs()).copy()
        ob += np.random.uniform(low=-0.1, high=0.1, size=ob.shape)

        return ob, reward, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        """
        dist_vec = obs[:, -3:]
        reward_dist = - np.linalg.norm(dist_vec, axis=1)
        reward_ctrl = - np.sum(np.square(acts), axis=1)
        reward = reward_dist + reward_ctrl

        # for the reward
        y, x, thetadot = np.cos(th), np.sin(th), thdot
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = y + .1 * x + .1 * (thetadot ** 2) + .001 * (u ** 2)
        reward = -costs

        def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

        """
        y, x, thetadot = obs[:, 0], obs[:, 1], obs[:, 2]
        u = np.clip(acts[:, 0], -self.max_torque, self.max_torque)
        costs = y + .1 * np.abs(x) + .1 * (thetadot ** 2) + .001 * (u ** 2)
        return -costs

    def tf_reward(self, obs, acts, next_obs):
        y, x, thetadot = obs[:, 0], obs[:, 1], obs[:, 2]
        u = tf.clip_by_value(acts[:, 0], -self.max_torque, self.max_torque)
        costs = y + .1 * tf.abs(x) + .1 * tf.square(thetadot) + .001 * tf.square(u)
        return -costs


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
