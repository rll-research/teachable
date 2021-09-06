# Allow us to interact wth the D4RLEnv the same way we interact with the TeachableRobotLevels class.
import numpy as np
import gym
from gym.spaces import Box
from d4rl_content.pointmaze.waypoint_controller import WaypointController
from envs.d4rl.oracle.batch_teacher import BatchTeacher
from oracle.cardinal_teacher import CardinalCorrections
from oracle.direction_teacher import DirectionCorrections
from oracle.waypoint_teacher import WaypointCorrections
from oracle.offset_waypoint_teacher import OffsetWaypointCorrections

from oracle.dummy_advice import DummyAdvice


class D4RLEnv:
    def __init__(self, env_name, offset_mapping=np.array([0, 0]), reward_type='dense', feedback_type=None,
                 max_grid_size=15, args=None, reset_target=True, reset_start=True, **kwargs):
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
        self.np_random = np.random.RandomState(kwargs.get('seed', 0))
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
        for ft in feedback_type:
            if ft == 'none': teachers[ft] = DummyAdvice()
            elif ft == 'Cardinal': teachers[ft] = CardinalCorrections()
            elif ft == 'Waypoint': teachers[ft] = WaypointCorrections()
            elif ft == 'OffsetWaypoint': teachers[ft] = OffsetWaypointCorrections()
            elif ft == 'Direction': teachers[ft] = DirectionCorrections()
        self.teacher = BatchTeacher(self.waypoint_controller, teachers)

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

    def state_to_waypoint_controller(self, state):
        grid = state['obs']
        state_len = int(len(grid) - (self.max_grid_size ** 2) / self.repeat_input)
        state = grid[:state_len]
        grid = grid[- (self.max_grid_size ** 2):]
        grid = grid.reshape(self.max_grid_size, self.max_grid_size)
        # Assume the current d4rl_env is the same size as the new maze
        h, w = self.waypoint_controller.env.gs.spec_no_start.shape
        grid = grid[: h, :w]
        pos = state[:2] * self.scale_factor
        target = state[-2:] * self.scale_factor
        # Assume om is the same (should be true for static envs, not necessarily for point_mass or for randommaze)
        if 'ant' in self.env_name:
            om = self._wrapped_env.env.wrapped_env._xy_to_rowcol(np.array([self._wrapped_env.env.wrapped_env._init_torso_x,
                                                                       self._wrapped_env.env.wrapped_env._init_torso_y]))
        else:
            om = np.array([0, 0])
        waypoint_controller = WaypointController(grid, offset_mapping=om)
        waypoint_controller.new_target(pos, target)

    def wall_distance(self):
        agent_pos = self.get_pos() + np.array(self.waypoint_controller.offset_mapping)
        agent_coord = np.round(agent_pos)
        x, y = agent_coord.astype(np.int32)
        x_pos, y_pos = agent_pos
        grid = self.waypoint_controller.env.gs.spec
        WALL = 0  # TODO: don't hardcode
        if x > 0 and grid[x - 1, y] == WALL:
            dist_to_wall_0 = x_pos - (x - 1)
        else:
            dist_to_wall_0 = 1
        if x < len(grid) - 1 and grid[x + 1, y] == WALL:
            dist_to_wall_1 = x + 1 - x_pos
        else:
            dist_to_wall_1 = 1
        if y > 0 and grid[x, y - 1] == WALL:
            dist_to_wall_2 = y_pos - (y - 1)
        else:
            dist_to_wall_2 = 1
        if y < len(grid) - 1 and grid[x, y + 1] == WALL:
            dist_to_wall_3 = y + 1 - y_pos
        else:
            dist_to_wall_3 = 1
        return np.array([dist_to_wall_0, dist_to_wall_1, dist_to_wall_2, dist_to_wall_3])

    def update_obs(self, obs_dict):
        state = self.waypoint_controller.env.gs.spec_no_start
        max_grid = np.zeros((self.max_grid_size, self.max_grid_size))
        h, w = state.shape
        max_grid[:h, :w] = state
        state_obs = obs_dict['obs']
        if self.args.env == 'ant':
            if self.args.show_pos == 'ours':
                state_obs[:2] = self.get_pos() / self.scale_factor
            elif self.args.show_pos == 'none':
                state_obs[:2] *= 0
            elif self.args.show_pos == 'default':
                state_obs[:2] /= 5
            if self.args.show_goal == 'ours':
                goal = self.get_target() / self.scale_factor
            elif self.args.show_goal == 'offset':
                goal = (self.get_target() - self.get_pos()) / self.scale_factor
            elif self.args.show_goal == 'none':
                goal = np.array([0, 0])
            state_obs = np.concatenate([state_obs, goal])
        obs_dict['obs'] = np.concatenate([state_obs] * self.repeat_input + [max_grid.flatten()])
        if self.teacher is not None and not 'None' in self.teacher.teachers:
            advice = self.teacher.give_feedback(self)
            obs_dict.update(advice)
        return obs_dict

    def get_success(self):
        target = self.get_target()
        agent_pos = self.get_pos()
        return np.linalg.norm(target - agent_pos) < .25

    def step(self, action):
        # action = np.tanh(action)
        action = np.clip(action, -1, 1)
        self.past_positions.append(self.get_pos())
        prev_pos = self.get_pos().copy()
        obs, rew, done, info = self._wrapped_env.step(action)
        obs = self.scale_obs(obs)
        self.waypoint_controller.new_target(self.get_pos(), self.get_target())
        # Distance to goal
        start_points = [self.get_pos()] + self.waypoint_controller.waypoints[:-1]
        end_points = self.waypoint_controller.waypoints
        distance = sum([np.linalg.norm(end - start) for start, end in zip(start_points, end_points)])
        gave_reward = True
        if self.reward_type == 'sparse':
            gave_reward = done
        if self.reward_type == 'oracle_action':
            act, _ = self.waypoint_controller.get_action(self.get_pos(), self.get_vel(), self.get_target())
            rew = -np.linalg.norm(action - act) / 100 + .03  # scale so it's not too big and is always positive
        elif self.reward_type == 'oracle_dist':
            rew = - distance / 100
        elif self.reward_type == 'waypoint':
            if len(self.waypoint_controller.waypoints) < self.min_waypoints:
                rew = 1
            else:
                rew = 0
            gave_reward = done or rew == 1
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
        elif self.reward_type == 'vector_next_waypoint':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            dir_desired = self.waypoint_controller.waypoints[0] - new_pos  # Same offset (or lack thereof)? Same scale?
            dir_desired = dir_desired / np.linalg.norm(dir_desired)
            rew = np.dot(dir_taken, dir_desired)
            if len(self.waypoint_controller.waypoints) < self.min_waypoints:
                rew += 1
        elif self.reward_type == 'wall_penalty':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            dir_desired = self.waypoint_controller.waypoints[0] - new_pos  # Same offset (or lack thereof)? Same scale?
            dir_desired = dir_desired / np.linalg.norm(dir_desired)
            rew = np.dot(dir_taken, dir_desired)
            agent_pos = new_pos + np.array(self.waypoint_controller.offset_mapping)
            agent_coord = np.round(agent_pos)
            x, y = agent_coord.astype(np.int32)
            x_pos, y_pos = agent_pos
            # compute the distance to all other cells
            grid = self.waypoint_controller.env.gs.spec
            WALL = 0  # TODO: don't hardcode
            dist_to_wall = 2
            if x > 0 and grid[x - 1, y] == WALL:
                dist_to_wall = min(dist_to_wall, x_pos - (x - 1))
            if x < len(grid) - 1 and grid[x + 1, y] == WALL:
                dist_to_wall = min(dist_to_wall, x + 1 - x_pos)
            if y > 0 and grid[x, y - 1] == WALL:
                dist_to_wall = min(dist_to_wall, y_pos - (y - 1))
            if y < len(grid[0]) - 1 and grid[x, y + 1] == WALL:
                dist_to_wall = min(dist_to_wall, y + 1 - y_pos)
            rew += dist_to_wall / 10000
            if len(self.waypoint_controller.waypoints) < self.min_waypoints:
                rew += 1
        elif self.reward_type == 'vector_dir_waypoint_negative':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            # dir_taken = dir_taken / np.linalg.norm(dir_taken)
            dir_desired = self.get_teacher_action()
            rew = np.dot(dir_taken, dir_desired)
            rew -= .5
            if len(self.waypoint_controller.waypoints) < self.min_waypoints:
                rew += 1
        elif self.reward_type == 'vector_dir_both':
            new_pos = self.get_pos().copy()
            dir_taken = new_pos - prev_pos
            # dir_taken = dir_taken / np.linalg.norm(dir_taken)
            dir_desired = self.get_teacher_action()
            rew = np.dot(dir_taken, dir_desired) / 5
            if len(self.waypoint_controller.waypoints) < self.min_waypoints:
                rew += 1
            success = self.get_success()
            if success:
                rew += 20  # TODO: is this a reasonable scale?
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
        info['gave_reward'] = gave_reward
        info['teacher_action'] = np.array(-1)
        info['episode_length'] = self._wrapped_env._elapsed_steps
        info['next_obs'] = obs_dict

        if hasattr(self, 'teacher') and self.teacher is not None:
            # Even if we use multiple teachers, presumably they all relate to one underlying path.
            # We can log what action is the next one on this path (currently in teacher.next_action).
            info['teacher_action'] = self.get_teacher_action()
            self.teacher.step(self)  # TODO: OBOE? should this be before update_obs?
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

    def scale_obs(self, obs):
        return obs

    def reset(self):
        self._wrapped_env = gym.envs.make(self.env_name, reset_target=self.reset_target, reset_start=self.reset_start,
                                          reward_type=self.reward_type)
        obs = self._wrapped_env.reset()
        obs = self.scale_obs(obs)
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
        # fake_target = np.random.randint(low=1, high=4, size=2)
        # target = fake_target / self.scale_factor
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
    
    def reset(self):
        obs_dict = super().reset()
        return obs_dict

    def step(self, action):
        obs_dict, rew, done, info = super().step(action)
        if self.reward_type == 'dense':
            rew = rew / 100 + .1
        return obs_dict, rew, done, info

    def scale_obs(self, obs):
        scale = 5
        obs[22] /= scale
        obs[24] /= scale
        obs[26] /= scale
        obs[28] /= scale
        return obs

class AntSACEnv(AntEnv):
    def step(self, action):
        obs_dict, rew, done, info = super().step(action)
        return obs_dict['obs'], rew, done, info

    def reset(self):
        obs_dict = super().reset()
        return obs_dict['obs']
