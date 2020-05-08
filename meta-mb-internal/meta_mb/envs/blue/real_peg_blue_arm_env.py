import numpy as np
import time
from meta_mb.logger import logger
import gym
from gym import error, spaces
from meta_mb.meta_envs.base import MetaEnv
from blue_interface.blue_interface import BlueInterface

class ArmPegInsertionEnv(MetaEnv, BlueInterface, gym.utils.EzPickle):
    def __init__(self, goal_dist=0.1, side='right', ip='127.0.0.1', port=9090):
        max_torques = np.array([5, 5, 4, 4, 3, 2, 2])
        self.frame_skip = 1
        self.dt = 0.2
        super(ArmPegInsertionEnv, self).__init__(side, ip, port)
        self.init_qpos = self.get_joint_positions()
        self._prev_qpos = self.init_qpos.copy()
        self.act_dim = len(max_torques)
        self.obs_dim = len(self._get_obs())
        self._low = -max_torques
        self._high = max_torques

        self.peg_loc = np.zeros(3)
        self.reach_goal = np.zeros(3)
        self.peg_board = np.zeros(3)
        self.reached = False
        self.goal_dist = goal_dist

        self.positions = {}
        self.actions = {}
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        #Still need to add the location of the peg and the distance
        return np.concatenate([
            self.get_joint_positions(),
            self.get_joint_velocities(),
            self.tip_position,
        ])

    def step(self, action):
        self._prev_qpos = self.get_joint_positions()
        self._prev_qvel = self.get_joint_velocities()
        self.do_simulation(act, self.frame_skip)

        vec = self.vec_gripper_to_goal

        joint_velocities = self.get_joint_velocities()

        if not self.reached:
            reward_dist = -1.5 * self.reach_dist()
        else:
            reward_dist = -self.peg_insertion()

        reward_ctrl = -np.square(joint_velocities)
        reward = reward_dist + 1.25e-8 * reward_ctrl

        ob = self._get_obs()

        done = False
        if self.actions is not None:
            action_num = len(self.actions)
            self.actions.update({action_num : act})
        if self.positiosn is not None:
            if len(self.positions) == 0:
                self.positions  = dict({0 : np.vstack((self._prev_qpos, self._prev_qvel))})
            else:
                pos_and_vel = np.vstack((self.get_joint_positions(), self.get_joint_velocities()))
                self.postions.update({len(self.positions) : pos_and_vel})

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def do_simulation(self, action, frame_skip):
        action = np.clip(action, self._low, self._high)
        assert frame_skip > 0:
        for _ in range(frame_skip):
            time.sleep(self.dt)
            self.set_joint_torques(action)

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            joint_velocities = self.sim.data.qvel
            reward_ctrl = -np.sum(np.square(joint_velocities), axis=1)
            if not self.reached:
                reward_dist = -1.5 * self.reach_dist()
            else:
                reward_dist = -self.peg_insertion()
            reward = reward_dist + 1.25e-8 * reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            reward_ctrl = -np.sum(np.square(act))
            reward_dist = -self.peg_dist()
            reward = reward_dist + 1.25e-4 * reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        else:
            raise NotImplementedError

    def reset(self):
        self.set_joint_positions(np.zeros(7), duration=5)
        while True:
            self.reach_goal = np.zeros(3) #CHANGE THIS
            self.peg_board = np.zeros(3) #CHANGE THIS

            if np.linalg.norm(self.reach_goal) < 2 or np.linalg.norm(self.peg_board) < 2:
                break
        return self._get_obs()

    def los_diagnostics(self, paths, prefix=''):
        dist = [-path["env_infos"]['reward_dist'] for path in paths]
        final_dist = [-path["env_infos"]['reward_dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDistance', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))

    def reach_dist(self):
        #return distance to point above peg hole
        return

    def peg_insertion(self):
        #return reward depending on if peg has been inserted
        return

    @property
    def tip_position(self):
        pose = self.get_cartesian_pose()
        return pose['position']

    @property
    def vec_gripper_to_goal(self):
        gripper_pos = self.tip_position
        if not self.reached:
            vec_gripper_to_goal = self.reach_goal - gripper_pos
        else:
            vec_gripper_to_goal = self.peg_board - gripper_pos
        return vec_gripper_to_goal

    @property
    def action_space(self):
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

if __name__ == "__main__":
    env = ArmPegInsertionEnv()
    while True:
        env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()

