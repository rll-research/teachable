import numpy as np
import gym
from gym import utils
from gym import spaces
from meta_mb.envs.cassie.assets.cassiemujoco import CassieSim, CassieVis, pd_in_t, state_out_t, CassieState
from meta_mb.logger import logger


# CASSIE_TORQUE_LIMITS = np.array([4.5*25, 4.5*25, 12.2*16, 12.2*16, 0.9*50]) # ctrl_limit * gear_ratio
# CASSIE_MOTOR_VEL_LIMIT = np.array([2900, 2900, 1300, 1300, 5500]) / 60 / (2*np.pi) # max_rpm / 60 / 2*pi
# P_GAIN_RANGE = [10, 10000]
# D_GAIN_RANGE = [1, 100]
# MODEL_TIMESTEP = 0.001
#
# DEFAULT_P_GAIN = 200
# DEFAULT_D_GAIN = 20
#
# NUM_QPOS = 34
# NUM_QVEL = 32
#
# CTRL_COST_COEF = 0.001
# STABILISTY_COST_COEF = 0.01


class CassieEnv(gym.Env, utils.EzPickle):
    def __init__(self, render=False, fix_pelvis=False, frame_skip=20,
                 stability_cost_coef=1e-2, ctrl_cost_coef=1.0, alive_bonus=0.5, impact_cost_coef=1e-5,
                 rotation_cost_coef=1e-2, policytask='balancing', ctrl_type='T', apply_forces=False):
        utils.EzPickle.__init__(**locals())
        self.sim = CassieSim()
        if render:
            self.vis = CassieVis()
        else:
            self.vis = None

        assert ctrl_type in ['T', 'P', 'V', 'TP', 'TV', 'PV', 'TPV']
        """
        * T: Torque ctrl        * TP: Torque + Position ctrl    * None or all: Torque + Position + Velocity
        * P: Positon ctrl       * TV: Torque + Velocity ctrl
        * V: Velocity ctrl      * PV: Position + Velocity ctr
        """

        self.parameters = {}
        self.set_default_parameters()
        self.fix_pelvis = fix_pelvis
        self.model_timestep = 0.01
        self.frame_skip = frame_skip
        self.task = policytask
        self.ctrl_type = ctrl_type
        self._pd_params_to_set = []
        self.apply_forces = apply_forces

        # action and observation space specs
        self.act_limits_array = self._build_act_limits_array()
        self.act_dim = self.act_limits_array.shape[0]

        self.num_qpos = self.parameters['num_qpos']
        self.num_qvel = self.parameters['num_qvel']

        # self.obs_dim = 35
        # self.obs_dim = 38
        self.obs_dim = 35#39# #42 39 36
        self.apply_random_force_counter = 0
        self.apply_random_force_maxcount = 0.1/(0.0005*frame_skip)
        self.prob_random_force = 0.15
        self.random_force_on = False
        self.epmaxlen = int(1000*20./frame_skip)

        # reward function coeffs
        self.stability_cost_coef = stability_cost_coef
        self.ctrl_cost_coef = ctrl_cost_coef
        self.impact_cost_coef = impact_cost_coef
        self.alive_bonus = alive_bonus
        self.rotation_cost_coef = rotation_cost_coef
        self._time_step = 0

        if fix_pelvis: self.sim.hold()

    @staticmethod
    def _cassie_state_to_obs(int_state, state):
        # pelvis
        pelvis_ori = np.array(int_state.pelvis.orientation).astype(np.float64)
        pelvis_rot_vel = np.array(int_state.pelvis.rotationalVelocity).astype(np.float64)
        pelvis_pos = np.array(int_state.pelvis.position)
        pelvis_transl_vel = np.array(int_state.pelvis.translationalVelocity).astype(np.float64)
        pelvis_trans_acc = np.array(int_state.pelvis.translationalAcceleration)

        # joints
        joint_pos = np.array(int_state.joint.position).astype(np.float64)
        joint_vel = np.array(int_state.joint.velocity).astype(np.float64)

        # motors
        motor_pos = np.array(int_state.motor.position).astype(np.float64)
        motor_vel = np.array(int_state.motor.velocity).astype(np.float64)

        # All qpos and all qvel
        qpos_filtered = np.array([motor_pos[0], motor_pos[1], motor_pos[2], motor_pos[3], joint_pos[0],
                                  joint_pos[1], motor_pos[4], motor_pos[5], motor_pos[6], motor_pos[7],
                                  motor_pos[8], joint_pos[3], joint_pos[4], motor_pos[9]])

        qvel_filtered = np.array([motor_vel[0], motor_vel[1], motor_vel[2], motor_vel[3], joint_vel[0],
                                  joint_vel[1], motor_vel[4], motor_vel[5], motor_vel[6], motor_vel[7],
                                  motor_vel[8], joint_vel[3], joint_vel[4], motor_vel[9]])

        # Obs
        # Warning: It does not have the pelvis trans velocity
        obs_filtered = np.concatenate([pelvis_ori, qpos_filtered, pelvis_rot_vel, qvel_filtered])
        obs = obs_filtered.copy()

        # Noisy Obs
        # obs = obs + np.random.normal(0, 0.001, size=len(obs))

        return obs

    def step(self, action):
        assert action.ndim == 1 and action.shape == (self.act_dim,)
        u = self._action_to_pd_u(action)
        # if self.apply_forces and self._time_step % 10 == 0:
        if np.random.uniform(low=0, high=1.0) < self.prob_random_force:
            self.random_force_on = True

        if self.apply_forces and self.random_force_on:
            self.apply_random_force_counter += 1
            if self.apply_random_force_counter > self.apply_random_force_maxcount:
                self.random_force_on = False
                self.apply_random_force_counter = 0
            self.apply_random_force()

        state, internal_state = self.do_simulation(u, self.frame_skip)
        obs = self._cassie_state_to_obs(internal_state, state)

        reward, forward_vel, control_cost = self.reward(internal_state, state, action)

        self._time_step += 1
        done = self.done(state)
        info = {'forward_vel': forward_vel, 'control_cost': control_cost}

        return obs, reward, done, info

    def reset(self):
        self.sim = CassieSim()
        if self.fix_pelvis: self.sim.hold()

        # Initial state randomization
        qpos = self.sim.get_state().qpos()
        qvel = self.sim.get_state().qvel()

        qpos = qpos + np.random.uniform(low=-0.01, high=0.01, size=len(qpos))
        qvel = qvel + np.random.uniform(low=-0.001, high=0.001, size=len(qvel))
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        u = self._action_to_pd_u(np.zeros(self.act_dim,))
        internal_state = self.sim.step_pd(u)
        state = self.sim.get_state()
        self._time_step = 0
        self.apply_random_force_counter = 0
        self.random_force_on = False        
        return self._cassie_state_to_obs(internal_state, state)

    def do_simulation(self, u, n_frames):
        assert n_frames >= 1
        for _ in range(n_frames):
            internal_state_obj = self.sim.step_pd(u) # step_pd returns state_out_t structure -> however this structure is still not fully understood
        joint_state = self.sim.get_state() # get CassieState object
        return joint_state, internal_state_obj

    def done(self, state):
        pelvis_pos = np.array(state.qpos())
        return pelvis_pos[2] < 0.65 or self._time_step > self.epmaxlen

    def reward(self, internal_state, state, action):
        # reward fct
        qvel = np.array(state.qvel())
        pelvis_rot_vel = qvel[3:6]
        pelvis_transl_vel = qvel[:3]

        foot_forces = self.get_foot_forces(internal_state)
        motor_torques = _to_np(internal_state.motor.torque)
        forward_vel = pelvis_transl_vel[0]
        ctrl_cost = self.ctrl_cost_coef * 0.5 * np.mean(np.square(motor_torques/self.torque_limits))
        stability_cost = self.stability_cost_coef * 0.5 * np.mean(np.square(pelvis_transl_vel[1:]))
        rotation_cost = self.rotation_cost_coef * 0.5 * np.mean(np.square(pelvis_rot_vel))
        impact_cost = self.impact_cost_coef * 0.5 * np.sum(np.square(np.clip(foot_forces, -1, 1)))

        if self.task == 'balancing':
            # vel_cost = self.stability_cost_coef * forward_vel ** 2
            # reward = - vel_cost - ctrl_cost - stability_cost - impact_cost + self.alive_bonus

            vel_cost = np.square(qvel[:6]).sum()
            ctrl_cost = 0.5 * np.mean(np.square(motor_torques/self.torque_limits))
            stability_cost = 0.5 * np.mean(np.square(qvel[:]))  #  quadratic velocity of pelvis in y and z direction ->
            impact_cost = 0.5 * np.sum(np.square(np.clip(foot_forces, -1, 1)))
            pelvis_pos = np.array(state.qpos())
            height_cost = 0.5 * np.sum(np.square(pelvis_pos[2] - 1.0))

            qpos_joints_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
            qpos_joints = np.asarray(state.qpos())[qpos_joints_idx]
            qpos_joints_ref = np.array([0, 0, 0.5, -1.2, -1.6, 0, 0, 0.5, -1.2, -1.6])
            jointref_cost = 0.5 * np.sum(np.square(qpos_joints - qpos_joints_ref))

            reward = 1.0 * np.exp(-100.0 * vel_cost) \
                     + 1.0 * np.exp(-100.0 * ctrl_cost) \
                     + 1.0 * np.exp(-10.0 * stability_cost) \
                     + 1.0 * np.exp(-100.0 * impact_cost) \
                     + 1.0 * float(pelvis_pos[2] > 0.9 and pelvis_pos[2] < 1.1) \
                     # + 6.0

                    # + 1.0*np.exp(-100.0 * height_cost) \
                    # + 1.0*np.exp(-100.0 * jointref_cost)

        elif self.task == 'fixed-vel':
            ctrl_cost = 0.5 * np.mean(np.square(motor_torques/self.torque_limits))
            impact_cost = 0.5 * np.sum(np.square(np.clip(foot_forces, -1, 1)))
            rotation_cost = 0.5 * np.mean(np.square(pelvis_rot_vel))

            reward = 1.0*float(forward_vel > 0.3 and forward_vel < 0.7) \
                    + 1.0*np.exp(-100.0 * ctrl_cost) \
                    + 1.0*np.exp(-100.0 * impact_cost) \
                    + 1.0*np.exp(-100.0 * rotation_cost)

            # vel_reward = np.exp(- (2.3 - forward_vel) ** 2)                    
            # reward = vel_reward - ctrl_cost - stability_cost - rotation_cost - impact_cost + self.alive_bonus
        else:
            ctrl_cost = 0.5 * np.mean(np.square(motor_torques/self.torque_limits))
            stability_cost = 0.5 * np.mean(np.square(qvel[:]))  #  quadratic velocity of pelvis in y and z direction ->
            impact_cost = 0.5 * np.sum(np.square(np.clip(foot_forces, -1, 1)))
            rotation_cost = 0.5 * np.mean(np.square(pelvis_rot_vel))

            reward = forward_vel \
                    + 1.0*np.exp(-100.0 * ctrl_cost) \
                    + 1.0*np.exp(-100.0 * impact_cost) \
                    + 1.0*np.exp(-100.0 * rotation_cost) \
                    + 1.0 * np.exp(-10.0 * stability_cost)

        return reward, forward_vel, ctrl_cost

    def render(self, *args, **kwargs):
        if self.vis is None:
            print('Setting up cassie visualizer')
            self.setup_cassie_vis()
        self.vis.draw(self.sim)

    def get_foot_forces(self, internal_state):
        left_toe = _to_np(internal_state.leftFoot.toeForce)
        left_heel = _to_np(internal_state.leftFoot.heelForce)
        right_toe = _to_np(internal_state.rightFoot.toeForce)
        right_heel = _to_np(internal_state.rightFoot.heelForce)
        return np.concatenate([left_toe, left_heel, right_toe, right_heel])

    def apply_random_force(self):
        force = np.zeros((6,))
        sample1 = np.random.choice([0, 10, 25, 50]) * np.random.choice([-1, 1])
        sample2 = np.random.choice([0, 10, 25, 50]) * np.random.choice([-1, 1])
        sample3 = -1 * np.random.choice([0, 10, 25, 50])
        force[0] = sample1
        force[1] = sample2
        force[2] = sample3
        self.sim.apply_force(force)

    @property
    def torque_limits(self):
        return np.concatenate([self.parameters['cassie_torque_limits']] * 2)

    @property
    def dt(self):
        return self.model_timestep

    @property
    def action_space(self):
        return spaces.Box(low=self.act_limits_array[:, 0], high=self.act_limits_array[:,1], dtype=np.float32)

    @property
    def observation_space(self):
        obs_limit = np.inf * np.ones(self.obs_dim)
        return spaces.Box(-obs_limit, obs_limit, dtype=np.float32)

    def setup_cassie_vis(self):
        self.vis = CassieVis()

    def _action_to_pd_u(self, action):
        """
        motors:
        0: hip abduction
        1: hip twist
        2: hip pitch -> lift leg up
        3: knee
        4: foot pitch

        Typical pGain ~ 200 [100, 10000]
        Typical dGain ~ 20
        Typical feedforward torque > 0
        """

        u = pd_in_t()
        act_idx = 0
        for leg_name in ['leftLeg', 'rightLeg']:
            leg = getattr(u, leg_name)
            for motor_id in range(5):
                for pd_param in ['torque', 'pTarget', 'dTarget', 'pGain', 'dGain']:
                    if pd_param in ['pGain', 'dGain']:
                        getattr(leg.motorPd, pd_param)[motor_id] = self.parameters[pd_param]
                    elif pd_param not in self._pd_params_to_set:
                        getattr(leg.motorPd, pd_param)[motor_id] = 0
                    else:
                        getattr(leg.motorPd, pd_param)[motor_id] = action[act_idx]
                        act_idx += 1
        assert act_idx == len(action)
        return u

    def _build_act_limits_array(self):
        limits = []
        p_gain, d_gain = 0, 0 # Put the gains to 0 if it isn't torque

        if 'T' in self.ctrl_type:
            self._pd_params_to_set.append('torque')
        if 'P' in self.ctrl_type:
            self._pd_params_to_set.append('pTarget')
            p_gain = self.parameters['pGain']
        if 'V' in self.ctrl_type:
            self._pd_params_to_set.append('dTarget')
            d_gain = self.parameters['dGain']

        self.parameters['pGain'], self.parameters['dGain'] = p_gain, d_gain

        for leg_name in ['leftLeg', 'rightLeg']:
            for motor_id in range(5):
                for pd_param in self._pd_params_to_set:
                    if pd_param == 'torque':
                        low, high = (-self.parameters['cassie_torque_limits'][motor_id],
                                     self.parameters['cassie_torque_limits'][motor_id])
                    elif pd_param == 'pTarget':
                        low, high = (-2 * np.pi, 2 * np.pi)
                    elif pd_param == 'dTarget':
                        low, high = (-self.parameters['cassie_motor_vel_limits'][motor_id],
                                     self.parameters['cassie_motor_vel_limits'][motor_id])
                    elif pd_param == 'pGain':
                        low, high = self.parameters['p_gain_range']
                    elif pd_param == 'dGain':
                        low, high = self.parameters['d_gain_range']
                    else:
                        raise AssertionError('Unknown pd_param %s' % pd_param)
                    limits.append(np.array([low, high]))
        limits_array = np.stack(limits, axis=0)
        assert limits_array.ndim == 2 and limits_array.shape[1] == 2
        return limits_array

    def set_default_parameters(self):
        self.parameters = dict(cassie_torque_limits=np.array([4.5*25, 4.5*25, 12.2*16, 12.2*16, 0.9*50]), # ctrl_limit * gear_ratio
                               cassie_motor_vel_limits=np.array([2900, 2900, 1300, 1300, 5500]) / 60 / (2 * np.pi), # max_rpm / 60 / 2*pi
                               p_gain_range=[10, 10000],
                               d_gain_range=[1, 100],
                               model_timestep=0.01, # TODO: See what this does
                               pGain=200,
                               dGain=20,
                               num_qpos=34,
                               num_qvel=32,
                               ctrl_cost_coef=0.001,
                               stability_cost_coef=0.01,)

    @staticmethod
    def log_diagnostics(paths, prefix=''):
        forward_vel = [np.mean(path['env_infos']['forward_vel']) for path in paths]
        ctrl_cost = [np.mean(path['env_infos']['control_cost']) for path in paths]
        # stability_cost = [np.mean(path['env_infos']['stability_cost']) for path in paths]
        path_length = [path["observations"].shape[0] for path in paths]

        logger.logkv(prefix + 'AvgForwardVel', np.mean(forward_vel))
        logger.logkv(prefix + 'StdForwardVel', np.std(forward_vel))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))
        # logger.logkv(prefix + 'AvgStabilityCost', np.mean(stability_cost))
        logger.logkv(prefix + 'AvgPathLength', np.mean(path_length))


def pelvis_height_from_obs(obs):
    if obs.ndim == 1:
        return obs[1]
    elif obs.ndim == 2:
        return obs[:, 1]
    else:
        raise NotImplementedError


def _to_np(o, dtype=np.float32):
    return np.array([o[i] for i in range(len(o))], dtype=dtype)


if __name__ == '__main__':
    render = True
    env = CassieEnv(render=render, fix_pelvis=False, frame_skip=200)
    while True:
        obs = env.reset()
        for j in range(1000):
            cum_forward_vel = 0
            act = env.action_space.sample()
            env.apply_random_force()
            obs, reward, done, info = env.step(act)
            if render:
                env.render()
