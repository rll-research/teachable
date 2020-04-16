import numpy as np
import time
from meta_mb.logger import logger
import gym
from gym import error, spaces
from meta_mb.meta_envs.base import MetaEnv

from blue_interface.blue_interface import BlueInterface

class ArmReacherEnv(MetaEnv, BlueInterface, gym.utils.EzPickle):
    def __init__(self, match_joints=False, fixed=False, side='right', ip='127.0.0.1', port=9090):
        self.goal = np.array([0.1, 0.21, -0.55])
        self.joint_goal_pos = np.zeros(7) # joint position control vs torque control
        self.match_joints = match_joints
        self.fixed = fixed
        max_torques = np.array([5, 5, 4, 4, 3, 2, 2]) #control 7 different joints not including the grippers
        self.frame_skip = 1
        self.dt = 0.2
        super(ArmReacherEnv, self).__init__(side, ip, port)
        self.init_qpos = self.get_joint_positions()
        self._prev_qpos = self.init_qpos.copy()
        self.act_dim = len(max_torques)
        self.obs_dim = len(self._get_obs())
        self._low = -max_torques
        self._high = max_torques
        self.positions = {}
        self.actions = {}
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            np.concatenate((self.get_joint_positions(), self.goal)),
            self.get_joint_velocities(),
            self.tip_position,
            self.vec_gripper_to_goal,
            ]).reshape(-1)

    def step(self, act):
        self._prev_qpos = self.get_joint_positions()
        self._prev_qvel = self.get_joint_velocities()
        if len(act) == 1:
            act = act[0]
        self.do_simulation(act, self.frame_skip)

        #self.correction() #FIXME
        if not self.match_joints:
            vec = self.vec_gripper_to_goal
            reward_dist = -np.linalg.norm(vec)
        else:
            reward = self.joint_match()

        reward_ctrl = -np.square(act/(2 * self._high)).sum()
        reward = reward_dist + 0.5 * 0.1 * reward_ctrl
        ob = self._get_obs()
        done = False

        if self.actions is not None:
            action_num = len(self.actions)
            self.actions.update({action_num : act})

        if self.positions is not None:
            if len(self.positions) == 0:
                self.positions = dict({0 : np.vstack((self._prev_qpos, self._prev_qvel))})
            else:
                arr = np.vstack((self.get_joint_positions(), self.get_joint_velocities()))
                self.positions.update({len(self.positions) : arr})

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def do_simulation(self, action,frame_skip):
        action = np.clip(action, self._low, self._high)
        assert frame_skip > 0
        for _ in range(frame_skip):
            time.sleep(self.dt)
            self.set_joint_torques(action)

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act / (2 * self._high)), axis=1)
            if self.match_joints:
                reward_dist = -np.linalg.norm(self.joint_goal())
            else:
                reward_dist = -np.linalg.norm(obs_next[:, -3:], axis=1)
            reward = reward_dist + reward_ctrl
            return np.clip(reward, -1e2, 1e2)

        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]

        else:
            raise NotImplementedError

    def reset(self):
        self.set_joint_positions(np.zeros(7), duration=5)
        while True:
            #self.goal = np.append(np.random.uniform(low=-0.75, hig-0.25), np.random.uniform(low=-0.2, high=0.2, size=3))
            if self.fixed and self.match_joints:
                self.joint_goal_pos = np.array([-np.pi/4.0, -0.78539816339, np.pi/4.0, 0, 0, 0, 0])
            elif not self.fixed and self.match_joints:
                self.joint_goal_pos = self.joint_goal()
            elif self.fixed:
                self.goal = np.array([-0.44, -0.1, 0.21])
            else:
                self.goal = np.random.uniform(low=[-0.75, -0.25, 0.25], high=[-0.25, 0.25, 0.5])
            if np.linalg.norm(self.goal) < 2:
                break
        return self._get_obs()

    def joint_goal(self):
        base_roll = np.random.uniform(low=-(np.pi/4.0), high=np.pi/4.0)
        right_shoulder_lift = np.random.uniform(low=-2.2944, high=-np.pi/4.0)
        right_shoulder_roll = np.random.uniform(low=-(np.pi/4.0), high=np.pi/4.0)
        right_elbow_lift = np.random.uniform(low=0, high=0)
        right_elbow_roll = np.random.uniform(low=0, high=0)
        right_wrist_lift = np.random.uniform(low=0, high=0)
        right_wrist_roll = np.random.uniform(low=0, high=0)
        return np.array([
            base_roll,
            right_shoulder_lift,
            right_shoulder_roll,
            right_elbow_lift,
            right_elbow_roll,
            right_wrist_lift,
            right_wrist_roll
        ])

    def joint_match(self):
        real_joint_positions = self.get_joint_positions()

        return -np.linalg.norm(real_joint_positions, self.joint_goal_pos)

    def correction(self):
        try:
            joints = self.get_joint_positions()
            joint_velocities = self.get_joint_velocities()
            shoulder_lift_joint = joints[1]
            if shoulder_lift_joint > -0.01:
                correction = -(0.5 * abs(shoulder_lift_joint)) + shoulder_lift_joint
            joints[1] = correction
            joint_velocities[1] = joint_velocities[1] * 0.9
            print("correction!")
            self.set_joint_positions(joints)
            self._joint_velocities = joint_velocities
        except:
            pass

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
        vec_arm_to_goal = self.goal_pos - arm_pos
        return vec_arm_to_goal
        

    @property
    def action_space(self):
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def log_diagnostics(self, paths, prefix=''):
        dist = [-path["env_infos"]['reward_dist'] for path in paths]
        final_dist = [-path["env_infos"]['reward_dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDistance', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))


if __name__ == "__main__":
    env = ArmReacherEnv(match_joints=True)
    while True:
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
            env.render()
