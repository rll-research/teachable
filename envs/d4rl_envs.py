# Allow us to interact wth the D4RLEnv the same way we interact with the TeachableRobotLevels class.
import numpy as np
import gym
import d4rl_content
import torch
from gym.spaces import Box, Discrete
from d4rl_content.pointmaze.waypoint_controller import WaypointController
from d4rl.oracle.batch_teacher import BatchTeacher
from oracle.cardinal_teacher import CardinalCorrections
from oracle.direction_teacher import DirectionCorrections
from oracle.waypoint_teacher import WaypointCorrections
from oracle.offset_waypoint_teacher import OffsetWaypointCorrections

from oracle.dummy_advice import DummyAdvice


class PointMassEnvSimple:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, env_name, feedback_type=None, feedback_freq=False, intermediate_reward=False,
                 cartesian_steps=[1], **kwargs):
        self.timesteps = 0
        self.time_limit = 10
        self.target = np.array([0, 0], dtype=np.float32)
        self.pos = np.array([3, 4], dtype=np.float32)
        self.feedback_type = feedback_type
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))  # TODO: seed isn't passed in
        self.teacher_action = np.array(-1)
        self.observation_space = Box(low=np.array([-5, -5]), high=np.array([5, 5]))
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]))

    def seed(self, *args, **kwargs):
        pass

    def step(self, action):
        action = np.clip(action, -1, 1)
        if action.shape == (1, 2):
            action = action[0]
        self.pos += action
        rew = -np.linalg.norm(self.target - self.pos) / 10
        self.timesteps += 1
        done = self.timesteps >= self.time_limit
        obs = self.pos
        obs_dict = {'obs': obs}
        reached_goal = np.linalg.norm(self.target - self.pos) < .49
        success = done and reached_goal
        info = {}
        info['success'] = success
        info['gave_reward'] = True
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self.timesteps
        return obs_dict, rew, done, info

    def set_task(self, *args, **kwargs):
        pass  # for compatibility with babyai, which does set tasks

    def reset(self):
        self.pos = np.array([3, 4], dtype=np.float32)
        self.timesteps = 0
        obs_dict = {'obs': self.pos}
        return obs_dict

    def render(self, mode='human'):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[48:52, 48:52, :2] = 1
        y = int(min(98, max(2, np.round(self.pos[0] * 10) + 50)))
        x = int(min(98, max(2, np.round(self.pos[1] * 10) + 50)))
        img[y - 2: y + 2, x - 2: x + 2] = 1
        return img * 255

    def vocab(self):  # We don't have vocab
        return [0]


class PointMassEnvSimpleDiscrete:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, env_name, feedback_type=None, feedback_freq=False, intermediate_reward=False,
                 cartesian_steps=[1], **kwargs):
        self.timesteps = 0
        self.time_limit = 10
        self.target = np.array([0, 0], dtype=np.float32)
        self.pos = np.array([3, 4], dtype=np.float32)
        self.feedback_type = feedback_type
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))  # TODO: seed isn't passed in
        self.teacher_action = np.array(-1)
        self.observation_space = Box(low=np.array([-5, -5]), high=np.array([5, 5]))
        self.action_space = Discrete(5)
        # TODO: create teachers

    def seed(self, *args, **kwargs):
        pass

    def step(self, action):
        if action == 0:
            action = np.array([-1, 0])
        elif action == 1:
            action = np.array([1, 0])
        elif action == 2:
            action = np.array([0, -1])
        elif action == 3:
            action = np.array([0, 1])
        elif action == 4:
            action = np.array([0, 0])
        else:
            print("uh oh")
        self.pos += action
        rew = -np.linalg.norm(self.target - self.pos) / 10
        self.timesteps += 1
        done = self.timesteps >= self.time_limit
        obs = self.pos
        obs_dict = {'obs': obs}
        success = done and np.linalg.norm(self.target - self.pos) < .49
        info = {}
        info['success'] = success
        info['gave_reward'] = True
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self.timesteps
        return obs_dict, rew, done, info

    def set_task(self, *args, **kwargs):
        pass  # for compatibility with babyai, which does set tasks

    def reset(self):
        self.pos = np.array([3, 4], dtype=np.float32)
        self.timesteps = 0
        obs_dict = {'obs': self.pos}
        return obs_dict

    def render(self, mode='human'):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[48:52, 48:52, :2] = 1
        y = int(min(98, max(2, np.round(self.pos[0] * 10) + 50)))
        x = int(min(98, max(2, np.round(self.pos[1] * 10) + 50)))
        img[y - 2: y + 2, x - 2: x + 2] = 1
        return img * 255

    def vocab(self):  # We don't have vocab
        return [0]


class D4RLEnv:
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, env_name, offset_mapping=np.array([0, 0]), reward_type='dense', feedback_type=None, feedback_freq=False,
                 cartesian_steps=[1], max_grid_size=15, args=None, reset_target=True, reset_start=True, **kwargs):
        self.env_name = env_name
        self.max_grid_size = max_grid_size
        self.reward_type = reward_type
        self.offset_mapping = offset_mapping
        self.args = args
        self.steps_since_recompute = 0
        self.past_positions = []
        self.past_imgs = []
        self.reset_target = reset_target
        self.reset_start = reset_start
        self._wrapped_env = gym.envs.make(env_name, reset_target=reset_target, reset_start=reset_start,
                                          reward_type=reward_type)
        self.feedback_type = feedback_type
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))  # TODO: seed isn't passed in
        self.teacher_action = self.action_space.sample() * 0 - 1
        if 'ant' in env_name:
            om = self._wrapped_env.env.wrapped_env._xy_to_rowcol(np.array([self._wrapped_env.env.wrapped_env._init_torso_x,
                                                                       self._wrapped_env.env.wrapped_env._init_torso_y]))
        else:
            om = np.array([0, 0])
        self.waypoint_controller = WaypointController(self.get_maze(), offset_mapping=om)
        self.scale_factor = 5
        self.repeat_input = 5
        teachers = {}
        if type(cartesian_steps) is int:
            cartesian_steps = [cartesian_steps]
        assert len(cartesian_steps) == 1 or len(cartesian_steps) == len(feedback_type), \
            "you must provide either one cartesian_steps value for all teachers or one per teacher"
        assert len(feedback_freq) == 1 or len(feedback_freq) == len(feedback_type), \
            "you must provide either one feedback_freq value for all teachers or one per teacher"
        if len(cartesian_steps) == 1:
            cartesian_steps = [cartesian_steps[0]] * len(feedback_type)
        if len(feedback_freq) == 1:
            feedback_freq = [feedback_freq[0]] * len(feedback_type)
        for ft, ff, cs in zip(feedback_type, feedback_freq, cartesian_steps):
            if ft == 'none':
                teachers[ft] = DummyAdvice(self, feedback_frequency=ff, cartesian_steps=cs,
                                                   controller=self.waypoint_controller)
            elif ft == 'Cardinal':
                teachers[ft] = CardinalCorrections(self, feedback_frequency=ff, cartesian_steps=cs,
                                                   controller=self.waypoint_controller)
            elif ft == 'Waypoint':
                teachers[ft] = WaypointCorrections(self, feedback_frequency=ff, cartesian_steps=cs,
                                                   controller=self.waypoint_controller)
            elif ft == 'OffsetWaypoint':
                teachers[ft] = OffsetWaypointCorrections(self, feedback_frequency=ff, cartesian_steps=cs,
                                                   controller=self.waypoint_controller)
            elif ft == 'Direction':
                teachers[ft] = DirectionCorrections(self, feedback_frequency=ff, cartesian_steps=cs,
                                                    controller=self.waypoint_controller)
        teacher = BatchTeacher(teachers)
        self.teacher = teacher
        # TODO: create teachers

    def get_timestep(self):
        return .02

    def get_target(self):
        raise NotImplementedError

    def get_pos(self):
        raise NotImplementedError

    def get_vel(self):
        raise NotImplementedError

    def get_maze(self):
        raise NotImplementedError

    def update_obs(self, obs_dict):
        state = self.waypoint_controller.env.gs.spec_no_start
        max_grid = np.zeros((self.max_grid_size, self.max_grid_size))
        h, w = state.shape
        max_grid[:h, :w] = state
        state_obs = obs_dict['obs'] / self.scale_factor
        obs_dict['obs'] = np.concatenate([state_obs] * self.repeat_input + [max_grid.flatten() / 5])  # TODO: /5 is a hacky way of trying to make the max grid less useful
        if self.teacher is not None and not 'None' in self.teacher.teachers:
            advice = self.teacher.give_feedback(self)
            obs_dict.update(advice)
        return obs_dict

    def get_success(self):
        target = self.get_target()
        agent_pos = self.get_pos()
        return np.linalg.norm(target - agent_pos) < .25

    def step(self, action):
        action = np.tanh(action)
        # action = np.clip(action, -1, 1)
        self.past_positions.append(self.get_pos())
        prev_pos = self.get_pos().copy()
        obs, rew, done, info = self._wrapped_env.step(action)
        self.waypoint_controller.new_target(self.get_pos(), self.get_target())
        # Distance to goal
        start_points = [self.get_pos()] + self.waypoint_controller.waypoints[:-1]
        end_points = self.waypoint_controller.waypoints
        distance = sum([np.linalg.norm(end - start) for start, end in zip(start_points, end_points)])

        if self.reward_type == 'oracle_action':
            act, _ = self.waypoint_controller.get_action(self.get_pos(), self.get_vel(), self.get_target())
            rew = -np.linalg.norm(action - act) / 100 + .03  # scale so it's not too big and is always positive
        elif self.reward_type == 'oracle_dist':
            rew = - distance / 100
        elif self.reward_type == 'vector_dir':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            # dir_taken = dir_taken / np.linalg.norm(dir_taken)
            dir_desired = self.get_teacher_action()
            rew = np.dot(dir_taken, dir_desired)
        elif self.reward_type == 'vector_dir_final':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            # dir_taken = dir_taken / np.linalg.norm(dir_taken)
            dir_desired = self.get_teacher_action()
            rew = np.dot(dir_taken, dir_desired)
            print("rew", rew)
            success = self.get_success()
            if success:
                rew += 5  # TODO: is this a reasonable scale?
        elif self.reward_type == 'vector_dir_waypoint':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            # dir_taken = dir_taken / np.linalg.norm(dir_taken)
            dir_desired = self.get_teacher_action()
            rew = np.dot(dir_taken, dir_desired)
            if len(self.waypoint_controller.waypoints) < self.min_waypoints:
                rew += 1
        elif self.reward_type == 'vector_dir_both':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            # dir_taken = dir_taken / np.linalg.norm(dir_taken)
            dir_desired = self.get_teacher_action()
            rew = np.dot(dir_taken, dir_desired)
            if len(self.waypoint_controller.waypoints) < self.min_waypoints:
                rew += 1
            success = self.get_success()
            if success:
                rew += 5  # TODO: is this a reasonable scale?
        elif self.reward_type == 'vector_dir2':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            # dir_taken = dir_taken / np.linalg.norm(dir_taken)  # normalize
            dir_desired = self.waypoint_controller.waypoints[0] - prev_pos
            dir_desired = dir_desired / np.linalg.norm(dir_desired)  # normalize
            rew = np.dot(dir_taken, dir_desired)
        self.min_waypoints = min(self.min_waypoints, len(self.waypoint_controller.waypoints))
        obs_dict = {}
        obs_dict["obs"] = obs
        obs_dict = self.update_obs(obs_dict)
        success = self.get_success()
        done = done or success
        self.done = done

        info = {}
        info['dist_to_goal'] = distance
        info['success'] = success
        info['gave_reward'] = True
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self._wrapped_env._elapsed_steps

        if hasattr(self, 'teacher') and self.teacher is not None:
            # Even if we use multiple teachers, presumably they all relate to one underlying path.
            # We can log what action is the next one on this path (currently in teacher.next_action).
            info['teacher_action'] = self.get_teacher_action()
            self.teacher.step(self)  # TODO: OBOE? should this be before update_obs?
            info['teacher_error'] = float(self.teacher.get_last_step_error())
            # Update the observation with the teacher's new feedback
            self.teacher_action = self.get_teacher_action()
        return obs_dict, rew, done, info

    def get_teacher_action(self):
        if hasattr(self, 'teacher') and self.teacher is not None:
            # Even if we use multiple teachers, presumably they all relate to one underlying path.
            # We can log what action is the next one on this path (currently in teacher.next_action).
            # Sanity check that all teachers have the same underlying path
            first_action = list(self.teacher.teachers.values())[0].next_action
            for teacher_name, teacher in self.teacher.teachers.items():
                if not np.array_equal(first_action, teacher.next_action):
                    print(f"Teacher Actions didn't match {[(k, int(v.next_action)) for k,v in self.teacher.teachers.items()]}")
            return list(self.teacher.teachers.values())[0].next_action
        return None

    def set_task(self, *args, **kwargs):
        pass  # for compatibility with babyai, which does set tasks

    def reset(self):
        self._wrapped_env = gym.envs.make(self.env_name, reset_target=self.reset_target, reset_start=self.reset_start,
                                          reward_type=self.reward_type)
        obs = self._wrapped_env.reset()
        obs_dict = {'obs': obs}
        self.steps_since_recompute = 0
        self.waypoint_controller.set_maze(self.get_maze())
        if 'ant' in self.env_name:
            om = self._wrapped_env.env.wrapped_env._xy_to_rowcol(np.array([self._wrapped_env.env.wrapped_env._init_torso_x,
                                                                       self._wrapped_env.env.wrapped_env._init_torso_y]))
        else:
            om = np.array([0, 0])
        self.waypoint_controller.offset_mapping = om
        self.waypoint_controller.new_target(self.get_pos(), self.get_target())
        self.min_waypoints = len(self.waypoint_controller.waypoints)
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher.reset(self)
        self.teacher_action = self.get_teacher_action()
        obs_dict = self.update_obs(obs_dict)
        self.past_positions = []
        self.past_imgs = []
        return obs_dict

    def vocab(self):  # We don't have vocab
        return [0]

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        # """
        try:
            if attr == '__len__':
                return None
            results = self.__getattribute__(attr)
            return results
        except:
            orig_attr = self._wrapped_env.__getattribute__(attr)

            if callable(orig_attr):
                def hooked(*args, **kwargs):
                    result = orig_attr(*args, **kwargs)
                    return result

                return hooked
            else:
                return orig_attr


class PointMassEnv(D4RLEnv):
    def __init__(self, *args, **kwargs):
        super(PointMassEnv, self).__init__(*args, offset_mapping=np.array([0, 0]), **kwargs)
        # Adding goal
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(6 * self.repeat_input + self.max_grid_size ** 2,))

    def get_target(self):
        return self._wrapped_env.get_target()

    def get_maze(self):
        return self._wrapped_env.get_maze()

    def get_pos(self):
        return self._wrapped_env.get_sim().data.qpos

    def get_vel(self):
        return self._wrapped_env.get_sim().data.qvel

    def step(self, action):
        obs_dict, rew, done, info = super().step(action)
        target = self.get_target() / self.scale_factor
        obs_dict['obs'] = np.concatenate([obs_dict['obs']] + [target] * self.repeat_input)
        if self.reward_type == 'dense':
            rew = rew / 10 - .01
        # done = done or info['success']
        if info['success']:
            rew += 1
        return obs_dict, rew, done, info

    def reset(self):
        obs_dict = super().reset()
        target = self.get_target() / self.scale_factor
        obs_dict['obs'] = np.concatenate([obs_dict['obs']] + [target] * self.repeat_input)
        return obs_dict


class PointMassSACEnv(PointMassEnv):
    def step(self, action):
        obs_dict, rew, done, info = super().step(action)
        return obs_dict['obs'], rew, done, info

    def reset(self):
        obs_dict = super().reset()
        return obs_dict['obs']



class AntEnv(D4RLEnv):
    def __init__(self, *args, **kwargs):
        super(AntEnv, self).__init__(*args, offset_mapping=np.array([1, 1]), **kwargs)
        size = len(self._wrapped_env.observation_space.low)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(size * self.repeat_input
                                                                                  + self.max_grid_size ** 2,))

    def get_target(self):
        return np.array(self._wrapped_env.xy_to_rowcolcontinuous(self._wrapped_env.get_target()))

    def get_maze(self):
        return self._wrapped_env.get_maze()

    def get_pos(self):
        return np.array(self._wrapped_env.xy_to_rowcolcontinuous(self._wrapped_env.get_xy()))

    def get_vel(self):
        return np.array([0, 0])  # TODO: is there a better option?

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def step(self, action):
        obs_dict, rew, done, info = super().step(action)
        if self.reward_type == 'dense':
            rew = rew / 100 + .1
        return obs_dict, rew, done, info

class AntSACEnv(AntEnv):
    def step(self, action):
        obs_dict, rew, done, info = super().step(action)
        return obs_dict['obs'], rew, done, info

    def reset(self):
        obs_dict = super().reset()
        return obs_dict['obs']
