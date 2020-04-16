import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from meta_mb.meta_envs.base import RandomEnv
import os
from meta_mb.envs.blue.full_blue_env import FullBlueEnv
import time

class MimicBlueActEnv(FullBlueEnv):

	def __init__(self, max_path_len, parent=None, actions=None):

		self.max_path_len = max_path_len
		self.path_len = 0

		self.parent = parent
		if actions is not None:
			self.actions = actions
		print(self.actions)

		self.goal_right = self.parent.goal

		FullBlueEnv.__init__(self)

	def step(self, act):
		action = self.actions[self.path_len]
		self.sim.model.body_pos[-1] = self.parent.goal
		self.do_simulation(action, self.frame_skip)
		vec_right = self.ee_position('right') - self.goal_right
		reward_dist = -np.linalg.norm(vec_right)
		reward_ctrl = -np.square(action/(2* self._high)).sum()
		reward = reward_dist + 0.5 * 0.1 * reward_ctrl
		observation = self._get_obs()
		if (self.path_len == self.max_path_len):
			done = True
		else:
			done = False
		info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
		self.path_len += 1
		return observation, reward, done, info

	def do_simulation(self, action, frame_skip):
		action = np.clip(action, self._low, self._high)
		assert frame_skip > 0
		for _ in range(frame_skip):
			time.sleep(self.dt)
			#Use normalized full blue env

	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat,
			self.sim.data.qvel.flat[:-3],
			self.sim.data.body_xpos.flat[:3],
			self.ee_position('right') - self.goal_right,
		])

if __name__ == "__main__":
    env = MimicBlueActEnv(max_path_len=200)
    while True:
        env.reset()
        for _ in range(200):
            env.step(env.action_space.sample())
            env.render()
