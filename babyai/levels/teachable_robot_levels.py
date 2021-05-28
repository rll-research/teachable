import copy
import pickle as pkl

from babyai.levels.levelgen import RoomGridLevel, RejectSampling
from meta_mb.meta_envs.base import MetaEnv
from gym_minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX, COLOR_TO_IDX, TILE_PIXELS
from gym_minigrid.roomgrid import RoomGrid, Room
import numpy as np
from copy import deepcopy
from babyai.oracle.post_action_advice import PostActionAdvice
from babyai.oracle.pre_action_advice import PreActionAdvice
from babyai.oracle.pre_action_advice_bad1 import PreActionAdviceBad1
from babyai.oracle.pre_action_advice_multiple import PreActionAdviceMultiple
from babyai.oracle.pre_action_advice_multiple_copy import PreActionAdviceMultipleCopy
from babyai.oracle.pre_action_advice_repeated import PreActionAdviceMultipleRepeated
from babyai.oracle.pre_action_advice_repeated_index import PreActionAdviceMultipleRepeatedIndex
from babyai.oracle.cartesian_corrections import CartesianCorrections
from babyai.oracle.cartesian_corrections_repeated_index import CartesianCorrectionsRepeatedIndex
from babyai.oracle.subgoal_corrections import SubgoalCorrections
from babyai.oracle.subgoal_simple_corrections import SubgoalSimpleCorrections
from babyai.oracle.offset_corrections import OffsetCorrections
from babyai.oracle.offset_corrections_repeated_index import OFFIO
from babyai.oracle.offset_sparse import OFFSparse
from babyai.oracle.offset_sparse_random import OFFSparseRandom
from babyai.oracle.off_sparse_random_easy import OSREasy
from babyai.oracle.osr_mistaken import OSRMistaken
from babyai.oracle.osr_periodic_explicit import OSRPeriodicExplicit
from babyai.oracle.osr_periodic_implicit import OSRPeriodicImplicit
from babyai.oracle.xy_corrections import XYCorrections
from babyai.oracle.batch_teacher import BatchTeacher
from babyai.oracle.dummy_advice import DummyAdvice
from babyai.bot import Bot
from vis_mask_grid import VisMaskGrid


class Level_TeachableRobot(RoomGridLevel, MetaEnv):
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, start_loc='all',
                 include_holdout_obj=True, num_meta_tasks=2,
                 persist_agent=True, persist_goal=True, persist_objs=True,
                 feedback_type=None, feedback_always=False, feedback_freq=False, intermediate_reward=False,
                 cartesian_steps=[1], fully_observed=False, padding=False, args=None, static_env=False, **kwargs):
        """
        :param start_loc: which part of the grid to start the agent in.  ['top', 'bottom', 'all']
        :param include_holdout_obj: If true, uses all objects. If False, doesn't use grey objects or boxes
        :param persist_agent: Whether to keep agent position the same across runs within a meta-task
        :param persist_goal: Whether to keep the goal (i.e. textual mission string) the same across runs in a meta-task
        :param persist_objs: Whether to keep object positions the same across runs within a meta-task
        :param feedback_type: Type of teacher feedback, string
        :param feedback_always: Whether to give that feedback type every time (rather than just when the agent needs help)
        :param kwargs: Additional arguments passed to the parent class
        """
        assert start_loc in ['top', 'bottom', 'all']
        self.start_loc = start_loc
        self.static_env = static_env
        self.intermediate_reward = intermediate_reward
        self.include_holdout_obj = include_holdout_obj
        self.persist_agent = persist_agent
        self.persist_goal = persist_goal
        self.persist_objs = persist_objs
        self.num_meta_tasks = num_meta_tasks
        self.task = {}
        self.itr = 0
        self.feedback_type = feedback_type
        self.fully_observed = fully_observed
        self.padding = padding
        super().__init__(**kwargs)
        if feedback_type is not None:
            rng = np.random.RandomState()
            self.oracle = {}
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
                    teacher = DummyAdvice(Bot, self, fully_observed=fully_observed)
                elif ft == 'PostActionAdvice':
                    teacher = PostActionAdvice(Bot, self, feedback_always=feedback_always,
                                               feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)
                elif ft == 'PreActionAdvice':
                    teacher = PreActionAdvice(Bot, self, feedback_always=feedback_always,
                                              feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)
                elif ft == 'PreActionAdviceBad1':
                    teacher = PreActionAdviceBad1(Bot, self, feedback_always=feedback_always,
                                              feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)
                elif ft == 'PreActionAdvice2':
                    teacher = PreActionAdvice(Bot, self, feedback_always=feedback_always,
                                              feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)
                elif ft == 'PreActionAdviceMultiple1':
                    teacher = PreActionAdviceMultiple(Bot, self, feedback_always=feedback_always,
                                                      feedback_frequency=ff, cartesian_steps=cs,
                                                      fully_observed=fully_observed)
                elif ft == 'PreActionAdviceMultipleCopy':
                    teacher = PreActionAdviceMultipleCopy(Bot, self, feedback_always=feedback_always,
                                                      feedback_frequency=ff, cartesian_steps=cs,
                                                          fully_observed=fully_observed)
                elif ft == 'PreActionAdviceMultipleRepeated':
                    teacher = PreActionAdviceMultipleRepeated(Bot, self, feedback_always=feedback_always,
                                                      feedback_frequency=ff, cartesian_steps=cs,
                                                              fully_observed=fully_observed)
                elif ft == 'PreActionAdviceMultipleRepeatedIndex':
                    teacher = PreActionAdviceMultipleRepeatedIndex(Bot, self, feedback_always=feedback_always,
                                                              feedback_frequency=ff, cartesian_steps=cs,
                                                                   fully_observed=fully_observed)
                elif ft == 'PreActionAdviceMultiple':
                    teacher = PreActionAdviceMultiple(Bot, self, feedback_always=feedback_always,
                                                      feedback_frequency=ff, cartesian_steps=cs,
                                                      fully_observed=fully_observed)
                elif ft == 'CartesianCorrections':
                    obs_size = self.reset()['obs'].flatten().size
                    teacher = CartesianCorrections(Bot, self, obs_size=obs_size, feedback_always=feedback_always,
                                                   feedback_frequency=ff, cartesian_steps=cs,
                                                   fully_observed=fully_observed)
                elif ft == 'CCIO':
                    obs_size = self.reset()['obs'].flatten().size
                    teacher = CartesianCorrectionsRepeatedIndex(Bot, self, obs_size=obs_size,
                                                                feedback_always=feedback_always,
                                                   feedback_frequency=ff, cartesian_steps=cs,
                                                                fully_observed=fully_observed)
                elif ft == 'SubgoalCorrections':
                    teacher = SubgoalCorrections(Bot, self, feedback_always=feedback_always,
                                                 feedback_frequency=ff, cartesian_steps=cs,
                                                 fully_observed=fully_observed)
                elif ft == 'SubgoalSimple':
                    teacher = SubgoalSimpleCorrections(Bot, self, feedback_always=feedback_always,
                                                 feedback_frequency=ff, cartesian_steps=cs,
                                                       fully_observed=fully_observed)
                elif ft == 'OffsetCorrections':
                    teacher = OffsetCorrections(Bot, self, feedback_always=feedback_always,
                                                feedback_frequency=ff, cartesian_steps=cs,
                                                fully_observed=fully_observed)
                elif ft == 'OFFIO':
                    teacher = OFFIO(Bot, self, feedback_always=feedback_always,
                                                feedback_frequency=ff, cartesian_steps=cs,
                                    fully_observed=fully_observed)
                elif ft == 'OFFSparse':
                    teacher = OFFSparse(Bot, self, feedback_always=feedback_always,
                                                feedback_frequency=ff, cartesian_steps=cs,
                                        fully_observed=fully_observed)
                elif ft == 'OFFSparseRandom':
                    teacher = OFFSparseRandom(Bot, self, feedback_always=feedback_always,
                                                feedback_frequency=ff, cartesian_steps=cs,
                                              fully_observed=fully_observed)
                elif ft == 'OFFSparseRandom2':
                    teacher = OFFSparseRandom(Bot, self, feedback_always=feedback_always,
                                              feedback_frequency=ff, cartesian_steps=cs,
                                              fully_observed=fully_observed)
                elif ft == 'OSREasy':
                    teacher = OSREasy(Bot, self, feedback_always=feedback_always,
                                              feedback_frequency=ff, cartesian_steps=cs,
                                      fully_observed=fully_observed)
                elif ft == 'OSREasy2':
                    teacher = OSREasy(Bot, self, feedback_always=feedback_always,
                                      feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)
                elif ft == 'OSRMistaken':
                    teacher = OSRMistaken(Bot, self, feedback_always=feedback_always,
                                              feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)

                elif ft == 'OSRPeriodicExplicit':
                    teacher = OSRPeriodicExplicit(Bot, self, feedback_always=feedback_always,
                                              feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)
                elif ft == 'OSRPeriodicImplicit':
                    teacher = OSRPeriodicImplicit(Bot, self, feedback_always=feedback_always,
                                              feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)
                elif ft == 'XYCorrections':
                    teacher = XYCorrections(Bot, self, feedback_always=feedback_always,
                                            feedback_frequency=ff, cartesian_steps=cs, fully_observed=fully_observed)
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

    def sample_task(self):
        """
        Sample a level-specific task
        :return: Dictionary containing some subset of the keys: [mission (specifies the text instruction),
                 agent (specifies the agent starting position and directino),
                 objs (specifies all object locations, types, and colors)]
        """
        tries = 0
        while True:
            try:
                tries += 1
                # Reset the grid first
                RoomGrid._gen_grid(self, self.width, self.height)

                task = {}
                if self.persist_goal:
                    mission = self.make_mission()
                    task['mission'] = mission

                if self.persist_agent:
                    self.add_agent()
                    task['agent'] = {'agent_pos': self.agent_pos, 'agent_dir': self.agent_dir}

                if self.persist_objs:
                    objs = self.add_objs(mission["task"])
                    task['objs'] = objs

                # If we have placed all components in place, sanity check that the placement is valid.
                if self.persist_goal and self.persist_agent and self.persist_objs:
                    self.validate_instrs(mission['instrs'])

            except RecursionError as error:
                # self.render(mode="human")
                # print('Timeout during mission generation:', error)
                continue

            except RejectSampling as e:
                if tries > 1000:
                    print("ISSUE sampling", e, type(self))
                    raise RejectSampling
                continue
            break
        return task

    # Functions fo RL2
    def set_task(self, _=None):
        """
        Sets task dictionary. The parameter is a dummy passed in for compatibility with the normal RL2 set task function
        """
        if not self.static_env:
            self.task = self.sample_task()
        self.itr = 0

    def get_task(self):
        """
        :return: task of the meta-learning environment. Dictionary.
        """
        return self.task

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
        Prevent doors from closing once open.
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

        # First load anything provided in the task
        if 'agent' in self.task:
            self.agent_pos = self.task['agent']['agent_pos']
            self.agent_dir = self.task['agent']['agent_dir']
        if 'mission' in self.task:
            mission = self.task['mission']
        if 'objs' in self.task:
            objs, goal_objs = self.task['objs']
            self.objs = deepcopy(objs)
            self.prevent_teacher_errors(self.objs)
            self.goal_objs = deepcopy(goal_objs)
            self.place_in_grid(self.objs)

        # Now randomly sample any required information not in the task
        if not 'agent' in self.task:
            self.add_agent()
        if not 'mission' in self.task:
            mission = self.make_mission()
        if not 'objs' in self.task:
            objs, goal_objs = self.add_objs(mission["task"])
            self.prevent_teacher_errors(objs)
            self.objs = deepcopy(objs)
            self.goal_objs = deepcopy(goal_objs)

        # self.compute_obj_infos()
        self.instrs = mission['instrs']
        if goal_objs is not None:
            # Currently if there are multiple goal objects, we just pick the first one. # TODO: handle multiple goal objs. Some of the teachers need these.
            if isinstance(goal_objs, tuple):
                goal_obj = goal_objs[0]
            else:
                goal_obj = goal_objs
            self.obj_pos = goal_obj.cur_pos

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
        try:
            action = action[0]
        except:
            action = action  # TODO: sketchy

        # Off by one error potentially.
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
            info['teacher_action'] = np.array(first_teacher.next_action, dtype=np.int32)
            if hasattr(first_teacher, 'num_steps'):
                info['num_steps'] = first_teacher.num_steps
            original_oracle = pkl.loads(pkl.dumps(self.oracle))
            self.oracle = self.teacher.step(action, self.oracle)
            for k, v in self.teacher.success_check(obs['obs'], action, self.oracle).items():
                info[f'followed_{k}'] = v
            info['teacher_error'] = float(self.teacher.get_last_step_error())
            # Update the observation with the teacher's new feedback
            self.teacher_action = self.get_teacher_action()
        else:
            original_oracle = None
            info['teacher_action'] = np.array(self.action_space.n, dtype=np.int32)
        obs = self.gen_obs(oracle=original_oracle, generate_feedback=True, past_action=action)
        # Reward at the end scaled by 1000
        reward_total = rew * 1000
        if self.intermediate_reward:
            provided_reward = True
            reward_total += int(give_reward) * 100
        else:
            provided_reward = done
        rew = reward_total / 1000
        info['gave_reward'] = int(provided_reward)
        self.done = done
        return obs, rew, done, info

    def compute_give_reward(self, action):  # TODO: consider computing dense rewards as a dictionary too
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
            vis_mask=full_vis_mask,
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

    # def _gen_grid(self, width, height):
    #     # Create the grid
    #     self.grid = VisMaskGrid(width, height)
    #
    #     self.room_grid = []
    #
    #     # For each row of rooms
    #     for j in range(0, self.num_rows):
    #         row = []
    #
    #         # For each column of rooms
    #         for i in range(0, self.num_cols):
    #             room = Room(
    #                 (i * (self.room_size-1), j * (self.room_size-1)),
    #                 (self.room_size, self.room_size)
    #             )
    #             row.append(room)
    #
    #             # Generate the walls for this room
    #             self.grid.wall_rect(*room.top, *room.size)
    #
    #         self.room_grid.append(row)
    #
    #     # For each row of rooms
    #     for j in range(0, self.num_rows):
    #         # For each column of rooms
    #         for i in range(0, self.num_cols):
    #             room = self.room_grid[j][i]
    #
    #             x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
    #             x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)
    #
    #             # Door positions, order is right, down, left, up
    #             if i < self.num_cols - 1:
    #                 room.neighbors[0] = self.room_grid[j][i+1]
    #                 room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
    #             if j < self.num_rows - 1:
    #                 room.neighbors[1] = self.room_grid[j+1][i]
    #                 room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
    #             if i > 0:
    #                 room.neighbors[2] = self.room_grid[j][i-1]
    #                 room.door_pos[2] = room.neighbors[2].door_pos[0]
    #             if j > 0:
    #                 room.neighbors[3] = self.room_grid[j-1][i]
    #                 room.door_pos[3] = room.neighbors[3].door_pos[1]
    #
    #     # The agent starts in the middle, facing right
    #     self.agent_pos = (
    #         (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
    #         (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
    #     )
    #     self.agent_dir = 0
