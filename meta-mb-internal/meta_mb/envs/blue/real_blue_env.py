import numpy as np
import time
from meta_mb.logger import logger
import gym
from gym import error, spaces
from meta_mb.meta_envs.base import MetaEnv
from blue_interface.blue_interface import BlueInterface


class BlueReacherEnv(MetaEnv, BlueInterface, gym.utils.EzPickle):
    def __init__(self, side='right', ip='127.0.0.1', port=9090):
        self.goal = np.array([0.5, 0.41, 0.65])  #When setting a goal, (x, y, z) in mujoco is (-y, z, x) in real life
        self.goal_position = np.array([-1.0, -1.5, 1.5, 0, 0, 0, 0])
        max_torques = np.array([5, 5, 4, 3, 3, 2, 2]) # Note: Just using the first 5 joints
        self.frame_skip = 1
        #self.dt = 0.02
        self.dt = 0.2 #frequency adjustment
        super(BlueReacherEnv, self).__init__(side, ip, port)
        self.init_qpos = self.get_joint_positions()
        self._prev_qpos = self.init_qpos.copy()
        self.act_dim = len(max_torques)
        self.obs_dim = len(self._get_obs())
        self._low, self._high = -max_torques, max_torques
        self.positions = {}
        self.actions = {}
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        self._prev_qpos = self.get_joint_positions()
        self._prev_qvel = self.get_joint_velocities()
        if (len(action) == 1):
            action = action[0]
        self.do_simulation(action, self.frame_skip)
        #vec = self.vec_gripper_to_goal
        vec = self.vec_arm_to_goal_pos
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action/(2 * self._high)).sum()
        reward = reward_dist + 0.5 * 0.1 * reward_ctrl
        ob = self._get_obs()
        done = False

        if self.actions is not None:
            action_num = len(self.actions)
            self.actions.update({action_num : action})

        if self.positions is not None:
            if len(self.positions) == 0:
                self.positions = dict({0 : np.vstack((self._prev_qpos, self._prev_qvel))})
            else:
                arr = np.vstack((self.get_joint_positions(), self.get_joint_velocities()))
                self.positions.update({len(self.positions) : arr})
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def do_simulation(self, action, frame_skip):
        action = np.clip(action, self._low, self._high)
        assert frame_skip > 0
        for _ in range(frame_skip):
            time.sleep(self.dt)
            self.set_joint_torques(action)

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

    def reset(self):
        self.set_joint_positions(np.zeros((7,)), duration=5.)
        #self.goal = np.array([0.5, 0.41, 0.65])
        #self.goal_position = np.array([
        #    np.random.uniform(low=-1, high=2),
        #    np.random.uniform(low=-1.5, high=-2),
        #    np.random.uniform(low=-1.5, high=1.5),
        #    0, 0, 0, 0])
        while True:
            # self.goal = np.random.uniform(low=-.2, high=.2, size=3)
            #self.goal = np.array([0.5, 0.41, 0.65]) # Note: this is with fixed goal
            self.goal_position = np.array([
            np.random.uniform(low=-1, high=2),
            np.random.uniform(low=-1.5, high=-2),
            np.random.uniform(low=-1.5, high=1.5),
            0, 0, 0, 0])
            if np.linalg.norm(self.goal_position) < 2:
                break
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            np.concatenate((self.get_joint_positions(), self.goal)),
            self.get_joint_velocities(),
            self.tip_position,
            self.vec_gripper_to_goal,
            ]).reshape(-1)

    @property
    def tip_position(self):
        pose = self.get_cartesian_pose()
        return pose['position']

    @property
    def vec_gripper_to_goal(self):
        gripper_pos = self.tip_position
        vec_gripper_to_goal = self.goal - gripper_pos
        return vec_gripper_to_goal

    @property
    def vec_arm_to_goal_pos(self):
        arm_pos = self.get_joint_positions()
        vec_arm_to_goal = self.goal_position - arm_pos
        return vec_arm_to_goal
    

    def log_diagnostics(self, paths, prefix=''):
        dist = [-path["env_infos"]['reward_dist'] for path in paths]
        final_dist = [-path["env_infos"]['reward_dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDistance', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))

    @property
    def action_space(self):
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)


if __name__ == "__main__":
    env = BlueReacherEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()
