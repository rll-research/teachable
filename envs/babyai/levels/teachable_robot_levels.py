import copy
import pickle as pkl

import torch

from envs.babyai.levels.levelgen import RoomGridLevel
from gym_minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX, COLOR_TO_IDX, TILE_PIXELS
import numpy as np
from copy import deepcopy
from envs.babyai.oracle.pre_action_advice import PreActionAdvice
from envs.babyai.oracle.subgoal_simple_corrections import SubgoalSimpleCorrections
from envs.babyai.oracle.off_sparse_random_easy import OSREasy
from envs.babyai.oracle.osr_mistaken import OSRMistaken
from envs.babyai.oracle.osr_periodic_explicit import OSRPeriodicExplicit
from envs.babyai.oracle.osr_periodic_implicit import OSRPeriodicImplicit
from envs.babyai.oracle.xy_corrections import XYCorrections
from envs.babyai.oracle.batch_teacher import BatchTeacher
from envs.babyai.oracle.dummy_advice import DummyAdvice
from envs.babyai.bot import Bot


class Level_TeachableRobot(RoomGridLevel):
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, start_loc='all',
                 include_holdout_obj=True, feedback_type=(), feedback_freq=(1,),
                 fully_observed=False, padding=False, args=None, static_env=False, **kwargs):
        """
        :param start_loc: which part of the grid to start the agent in.  ['top', 'bottom', 'all']
        :param include_holdout_obj: If true, uses all objects. If False, doesn't use grey objects or boxes
        :param feedback_type: Type of teacher feedback, string
        :param kwargs: Additional arguments passed to the parent class  # TODO: add more!
        """
        assert start_loc in ['top', 'bottom', 'all']
        self.start_loc = start_loc
        self.static_env = static_env
        self.include_holdout_obj = include_holdout_obj
        self.task = {}
        self.itr = 0
        self.feedback_type = feedback_type
        self.fully_observed = fully_observed
        self.padding = padding
        self.args = args
        super().__init__(**kwargs)
        if feedback_type is not None:
            rng = np.random.RandomState()
            self.oracle = {}
            teachers = {}
            assert len(feedback_freq) == 1 or len(feedback_freq) == len(feedback_type), \
                "you must provide either one feedback_freq value for all teachers or one per teacher"
            if len(feedback_freq) == 1:
                feedback_freq = [feedback_freq[0]] * len(feedback_type)
            for ft, ff in zip(feedback_type, feedback_freq):
                if ft == 'none':
                    teacher = DummyAdvice(Bot, self, fully_observed=fully_observed)
                elif ft == 'PreActionAdvice':
                    teacher = PreActionAdvice(Bot, self, feedback_frequency=ff, fully_observed=fully_observed)
                elif ft == 'SubgoalSimple':
                    teacher = SubgoalSimpleCorrections(Bot, self, feedback_frequency=ff, fully_observed=fully_observed)
                elif ft == 'OSREasy':
                    teacher = OSREasy(Bot, self, feedback_frequency=ff, fully_observed=fully_observed)
                elif ft == 'OSRMistaken':
                    teacher = OSRMistaken(Bot, self, feedback_frequency=ff, fully_observed=fully_observed)
                elif ft == 'OSRPeriodicExplicit':
                    teacher = OSRPeriodicExplicit(Bot, self, feedback_frequency=ff, fully_observed=fully_observed)
                elif ft == 'OSRPeriodicImplicit':
                    teacher = OSRPeriodicImplicit(Bot, self, feedback_frequency=ff, fully_observed=fully_observed)
                elif ft == 'XYCorrections':
                    teacher = XYCorrections(Bot, self, feedback_frequency=ff, fully_observed=fully_observed)
                else:
                    raise NotImplementedError(ft)
                teachers[ft] = teacher
                self.oracle[ft] = Bot(self, rng=copy.deepcopy(rng), fully_observed=fully_observed)
            teacher = BatchTeacher(teachers)
        else:
            teacher = None
        self.teacher = teacher

    def get_full_observation(self):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        # If carrying, set the carried object at [0,0]
        if self.carrying:
            object_encoding = self.carrying.encode()
            full_grid[0, 0] = np.array([
                object_encoding[0],
                object_encoding[1],
                1,
            ])
        return full_grid

    def sample_object(self):
        """
        Choose a random object (not a door).  If specified, hold out the color grey and type box.
        :return: (object type, object color).  Both are strings.
        """
        if self.include_holdout_obj:
            color = self.np_random.choice(['red', 'green', 'blue', 'purple', 'yellow', 'grey'])
            obj_type = self.np_random.choice(['key', 'ball', 'box'])
        else:
            color = self.np_random.choice(['red', 'green', 'blue', 'purple', 'grey'])
            obj_type = self.np_random.choice(['key', 'box', 'ball'])
        return obj_type, color

    def get_timestep(self):
        return .25

    def make_mission(self):
        """
        Defines mission. Must be implemented in all child classes.
        :return: {'instrs': Instructions object, 'task': arbitrary object containing data relevant for the task}
        """
        raise NotImplementedError

    def add_objs(self, task):
        """
        Defines the positions, types, and colors of objects in the grid (inc doors).
        Must be implemented in all child classes.
        :param task: the task object output by the class's make_mission function
        :return: (list of all objects, goal object or tuple if multiple)
        """
        raise NotImplementedError

    def get_color_idx(self, color):
        """
        Get the index of a color
        :param color: color string
        :return: index (int)
        """
        return ['red', 'green', 'blue', 'purple', 'yellow', 'grey'].index(color)

    def get_type_idx(self, obj_type):
        """
        Get the index of an object type
        :param obj_type: object type string
        :return: index (int)
        """
        return ['door', 'key', 'ball', 'box', 'lava'].index(obj_type)

    def compute_obj_infos(self):
        """
        Convert object positions, types, colors into a vector. Always order by the object positions to stay consistent
        and so it can't actually memorize.
        :return: 1D array with concatenated data on all the objects in the grid.
        """
        dist_pos_unwrapped = []
        for obj in self.objs:
            i, j = obj.cur_pos
            # If it's [-1, -1] i.e. being carried, just return that
            if (i, j) == (-1, -1):
                dist_pos_unwrapped.append(-1)
            else:
                dist_pos_unwrapped.append(i * self.room_size + j)
        idx = range(len(dist_pos_unwrapped) + 1)
        self.idx = [x for _, x in sorted(zip(dist_pos_unwrapped, idx))]

        dist_colors = [self.get_color_idx(d.color) for d in self.objs]
        dist_types = [self.get_type_idx(d.type) for d in self.objs]

        self.obj_infos = np.concatenate([np.array(dist_colors)[self.idx],
                                         np.array(dist_types)[self.idx],
                                         np.array(dist_pos_unwrapped)[self.idx]])

    def add_agent(self):
        """
        Place the agent randomly into the grid.  Optionally place it into a specific half of the grid.
        """
        cutoff = int(self.room_size / 2)
        top_index = (0, cutoff) if self.start_loc == 'bottom' else (0, 0)
        bottom_index = (self.room_size, cutoff) if self.start_loc == 'top' else (self.room_size, self.room_size)
        self.place_agent(top_index=top_index, bottom_index=bottom_index)

    def place_in_grid(self, objs):
        """
        Place objects in a grid.  The objects contain their own positions
        :param objs: a list of objects
        """
        for obj in objs:
            i, j = obj.cur_pos
            try:
                self.put_obj(obj, i, j)
            except Exception as e:
                print(e)

    def prevent_teacher_errors(self, objs):
        """
        The teacher fails in certain situations.  Here we modify the objects in the environment to avoid this.
        Prevent boxes from disappearing. When a box disappears, it is replaced by its contents (or None, if empty.)
        :param objs:
        """
        for obj in objs:
            if obj.type == 'box':
                obj.contains = obj

    def gen_mission(self):
        """
        Generate the mission for a single meta-task.  Any environment setup elements in the self.task dictionary
        are loaded from there.  All others are randomly sampled.
        """
        self.add_agent()
        mission = self.make_mission()
        objs, goal_objs = self.add_objs(mission["task"])
        self.prevent_teacher_errors(objs)
        self.objs = deepcopy(objs)
        self.goal_objs = deepcopy(goal_objs)
        self.instrs = mission['instrs']


    def get_doors(self):
        """
        Get a list of all doors in the environment.
        :return: List of Door objects
        """
        doors = []
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)
        return doors


    def place_agent(self, i=None, j=None, rand_dir=True, top_index=None, bottom_index=None):
        """
        Place the agent randomly into a room.  Optionally, initialize it in a particular part of the room.
        :param i: x-index of the room the agent is in.  If not provided, it is sampled randomly.
        :param j: y-index of the room the agent is in.  If not provided, it is sampled randomly.
        :param rand_dir: boolean specifying whether the agent should face a random direction
        :param top_index: Index of the highest square within a room the agent can be placed in.  If not provided,
                          it can be placed arbitrarily high in the room.
        :param bottom_index: Index of the lowest square within a room the agent can be placed in.  If not provided,
                          it can be placed arbitrarily low in the room.
        :return: agent position.  self.agent_pos and self.agent_dir are also set.
        """

        if i is None:
            i = self._rand_int(0, self.num_cols)
        if j is None:
            j = self._rand_int(0, self.num_rows)

        room = self.room_grid[j][i]

        if top_index is None:
            top_index = room.top
        if bottom_index is None:
            bottom_index = room.size

        # Find a position that is not right in front of an object
        while True:
            MiniGridEnv.place_agent(self, top_index, bottom_index, rand_dir, max_tries=1000)
            front_cell = self.grid.get(*self.front_pos)
            if front_cell is None or front_cell.type is 'wall':
                break

        return self.agent_pos

    def vocab(self):
        """
        All possible words in instruction strings.
        :return: List of word strings
        """
        colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
        types = ['door', 'key', 'ball', 'box']
        actions = ["go", "pick", "up", "open", "put"]
        fillers = ["to", "next", "the", "a"]
        misc = ["object", "lime", "chest", "seek", "matching", "object", "then"]
        unknown = [f"unk{i}" for i in range(10)]
        return ['PAD'] + colors + types + actions + fillers + misc + unknown

    def to_vocab_index(self, mission, pad_length=None):
        """
        Take a mission string, and return a fixed-length vector where each index is the index of the nth word in the
        mission.  The end is padded with 0
        :param mission: mission text as a string
        :param pad_length: length to pad the mission string to
        :return: list of integer indices of length pad_length (or len(mission.split(" ")) if pad_length is not provided)
        """
        words = mission.replace(",", "").split(" ")
        vocab = self.vocab()
        try:
            mission_list = [vocab.index(word) for word in words]
        except:
            print("?", words, [word in vocab for word in words])
        if pad_length is not None:
            mission_list = mission_list + [0] * (pad_length - len(mission_list))
        if len(mission_list) > pad_length:
            raise ValueError("Mission is too long: " + mission + str(pad_length))
        return mission_list

    def gen_obs(self, oracle=None, past_action=None, generate_feedback=False):
        """
        Generate the agent's view (partially observable). It's a concatenation of the agent's direction and position,
        the flattened partially observable view in front of it, the encoded mission string, and the teacher's feedback,
        if available.
        :return: np array of the agent's observation
        """
        if self.padding:
            image = self.get_full_observation()
            h, w, c = image.shape
            image = np.rot90(image, k=self.agent_dir)
            if self.agent_dir == 0:
                x = self.agent_pos[0]
                y = self.agent_pos[1]
            elif self.agent_dir == 1:
                x = w - self.agent_pos[1] - 1
                y = self.agent_pos[0]
            elif self.agent_dir == 2:
                x = w - self.agent_pos[0] - 1
                y = h - self.agent_pos[1] - 1
            elif self.agent_dir == 3:
                x = self.agent_pos[1]
                y = h - self.agent_pos[0] - 1
            image = (image, x, y)
        elif self.fully_observed:
            image = self.get_full_observation()
        else:
            grid, vis_mask = self.gen_obs_grid()
            # Encode the partially observable view into a numpy array
            image = grid.encode(vis_mask)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        goal = self.to_vocab_index(self.mission, pad_length=15)
        obs_dict = {}
        additional = deepcopy(np.concatenate([[self.agent_dir], self.agent_pos]))
        obs_dict["obs"] = image
        obs_dict['instr'] = goal
        obs_dict['extra'] = additional
        if generate_feedback and hasattr(self, 'teacher') and self.teacher is not None and not 'None' in self.teacher.teachers:
            if oracle is None:
                oracle = self.oracle
            if past_action is None:
                past_action = self.get_teacher_action()
            correction = self.compute_teacher_advice(image, past_action, oracle)
            obs_dict.update(correction)
        return obs_dict

    def compute_teacher_advice(self, obs, next_action, oracle):
        if self.reset_yet is False:
            correction = self.teacher.empty_feedback()
        else:
            correction = self.teacher.give_feedback(obs, next_action, oracle)
        return correction

    def step(self, action):
        """
        Step both the env and the teacher
        :param action: action to take
        :return: results of stepping the env
        """
        if type(action) in [np.ndarray, torch.IntTensor]:
            action = action.item()
        action = int(action)

        # Off by one error potentially.  # TODO: double check!
        if hasattr(self, 'teacher') and self.teacher is not None:
            give_reward = self.compute_give_reward(action)
        else:
            give_reward = False

        obs, rew, done, info = super().step(action)
        info['agent_pos'] = self.agent_pos
        info['agent_dir'] = self.agent_dir
        info['agent_room'] = self.room_from_pos(*self.agent_pos).top
        info['step'] = self.itr

        if hasattr(self, "obj_pos"):
            info['goal_room'] = self.room_from_pos(*self.obj_pos).top
            info['goal_pos'] = self.obj_pos
        else:
            info['goal_room'] = (-1, -1)
            info['goal_pos'] = (-1, -1)

        if hasattr(self, 'teacher') and self.teacher is not None:
            # Even if we use multiple teachers, presumably they all relate to one underlying path.
            # We can log what action is the next one on this path (currently in teacher.next_action).
            first_teacher = list(self.teacher.teachers.values())[0]
            info['teacher_action'] = np.array([first_teacher.next_action], dtype=np.int32)
            if hasattr(first_teacher, 'num_steps'):
                info['num_steps'] = first_teacher.num_steps
            original_oracle = pkl.loads(pkl.dumps(self.oracle))
            self.oracle = self.teacher.step(action, self.oracle)
            for k, v in self.teacher.success_check(obs['obs'], action, self.oracle).items():
                info[f'followed_{k}'] = v
            # Update the observation with the teacher's new feedback
            self.teacher_action = self.get_teacher_action()
        else:
            original_oracle = None
            info['teacher_action'] = np.array(self.action_space.n, dtype=np.int32)
        obs = self.gen_obs(oracle=original_oracle, generate_feedback=True, past_action=action)
        info['next_obs'] = obs
        # Reward at the end scaled by 1000
        if self.args.reward_type == 'dense':
            provided_reward = True
            rew += int(give_reward) * .1
        elif self.args.reward_type == 'dense_pos_neg':
            provided_reward = True
            rew += .1 if give_reward else -.1
        elif self.args.reward_type == 'dense_success':
            provided_reward = True
            rew += int(give_reward) * .1
            if done:
                rew += 2
        elif self.args.reward_type == 'sparse':
            provided_reward = done
        else:
            raise NotImplementedError(f"unrecognized reward type {self.args.reward_type}")
        info['gave_reward'] = int(provided_reward)
        self.done = done
        return obs, rew, done, info

    def compute_give_reward(self, action):
        # Give reward whenever the agent follows the optimal action
        give_reward = action == self.teacher_action
        return give_reward

    def get_teacher_action(self):
        if hasattr(self, 'teacher') and self.teacher is not None:
            # Even if we use multiple teachers, presumably they all relate to one underlying path.
            # We can log what action is the next one on this path (currently in teacher.next_action).
            if isinstance(self.teacher, BatchTeacher):
                # Sanity check that all teachers have the same underlying path
                first_action = list(self.teacher.teachers.values())[0].next_action
                for teacher_name, teacher in self.teacher.teachers.items():
                    if not first_action == teacher.next_action:
                        print(f"Teacher Actions didn't match {[(k, int(v.next_action)) for k,v in self.teacher.teachers.items()]}")
                return np.array(list(self.teacher.teachers.values())[0].next_action, dtype=np.int32)
            else:
                return np.array(self.teacher.next_action, dtype=np.int32)
        return None

    def reset(self):
        """
        Reset both the env and the teacher
        :return: observation from the env reset
        """
        if self.static_env:
            self.seed(0)
        super().reset()
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.oracle = self.teacher.reset(self.oracle)

        self.teacher_action = self.get_teacher_action()
        obs = self.gen_obs(generate_feedback=True, past_action=-1)
        self.itr += 1
        return obs

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS, full_vis_mask=None):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
            # vis_mask=full_vis_mask,
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img
