import numpy as np 
from gym.envs.mujoco import mujoco_env
from gym import utils
from meta_mb.meta_envs.base import RandomEnv
import os
from meta_mb.envs.blue.blue_env import BlueEnv
import time

class MimicBluePosEnv(BlueEnv):

	def __init__(self, max_path_len, positions=None):
		self.max_path_len = max_path_len
		self.path_len = 0
		if positions is not None:
			self.positions = positions
		BlueEnv.__init__(self)

	def step(self, action):
		self.do_simulation(action, self.frame_skip)
		vec_right = self.ee_position - self.goal
		reward_dist = -np.linalg.norm(vec_right)
		reward_ctrl = -np.square(action/(2* self._high)).sum()
		reward = reward_dist + 0.5 * 0.1 * reward_ctrl
		observation = self._get_obs()
		if (self.path_len == self.max_path_len):
			done = True
		else:
			done = False
		info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
		return observation, reward, done, info

	def do_simulation(self, action, frame_skip):
		action = np.clip(action, self._low, self._high)
		assert frame_skip > 0
		if len(self.positions) != 0:
			position = self.positions[self.path_len]
		for _ in range(frame_skip):
			#time.sleep(self.dt)
			qpos = np.concatenate((position[0], self.goal))
			qvel = np.concatenate((position[1], np.zeros(3)))
			self.set_state(qpos, qvel)
		self.path_len += 1

	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat,
			self.sim.data.qvel.flat[:-3],
			self.ee_position,
			self.ee_position - self.goal,
		])

if __name__ == "__main__":
    env = MimicBluePosEnv(max_path_len=200)
    while True:
        env.reset()
        for _ in range(200):
            env.step(env.action_space.sample())
            env.render()