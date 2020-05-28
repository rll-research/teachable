from .verifier import *
from babyai.levels.levelgen import RoomGridLevel, RejectSampling
from meta_mb.meta_envs.base import MetaEnv
from gym_minigrid.minigrid import MiniGridEnv, Key, Ball, Box
from gym_minigrid.roomgrid import RoomGrid
import numpy as np
from copy import deepcopy
from babyai.oracle.post_action_advice import PostActionAdvice
from babyai.oracle.pre_action_advice import PreActionAdvice
from babyai.oracle.cartesian_corrections import CartesianCorrections

from babyai.bot import Bot

class Level_TeachableRobot(RoomGridLevel, MetaEnv):
    """
    Parent class to all of the BabyAI envs (TODO: except the most complex levelgen ones currently)
    Provides functions to use with meta-learning, including sampling a task and resetting the same task
    multiple times for multiple runs within the same meta-task.
    """

    def __init__(self, start_loc='all',
                 include_holdout_obj=True,
                 persist_agent=True, persist_goal=True, persist_objs=True,
                 dropout_goal=0, dropout_correction=0, dropout_independently=True, 
                 feedback_type=None, feedback_always=False, **kwargs):
        """
        :param start_loc: which part of the grid to start the agent in.  ['top', 'bottom', 'all']
        :param include_holdout_obj: If true, uses all objects. If False, doesn't use grey objects or boxes
        :param persist_agent: Whether to keep agent position the same across runs within a meta-task
        :param persist_goal: Whether to keep the goal (i.e. textual mission string) the same across runs in a meta-task
        :param persist_objs: Whether to keep object positions the same across runs within a meta-task
        :param dropout_goal: Proportion of the time to dropout the goal (i.e. textual mission string)
        :param dropout_correction: Proportion of time to dropout corrections (chosen independently from goal dropout)
        :param dropout_independently: Whether to sample instances of goal and correction dropout independently
               If False, it drops out the goal normally, then only drops out the correction if the goal isn't dropped
               out (so the probability that the correction is dropped out is actually (1 - goal_dropout) * corr_dropout
        :param feedback_type: Type of teacher feedback, string
        :param feedback_always: Whether to give that feedback type every time (rather than just when the agent needs help)
        :param kwargs: Additional arguments passed to the parent class
        """
        assert start_loc in ['top', 'bottom', 'all']
        self.start_loc = start_loc
        self.include_holdout_obj = include_holdout_obj
        self.persist_agent = persist_agent
        self.persist_goal = persist_goal
        self.persist_objs = persist_objs
        self.dropout_goal = dropout_goal
        self.dropout_correction = dropout_correction
        self.dropout_independently = dropout_independently
        self.task = {}
        self.itr = 0
        super().__init__(**kwargs)
        if feedback_type == 'PostActionAdvice':
            teacher = PostActionAdvice(Bot, self, feedback_always=feedback_always)
        elif feedback_type == 'PreActionAdvice':
            teacher = PreActionAdvice(Bot, self, feedback_always=feedback_always)
        elif feedback_type == 'CartesianCorrections':
            teacher = CartesianCorrections(Bot, self, feedback_always=feedback_always)
        else:
            teacher = None
        self.teacher = teacher

    def sample_object(self):
        """
        Choose a random object (not a door).  If specified, hold out the color grey and type box.
        :return: (object type, object color).  Both are strings.
        """
        if self.include_holdout_obj:
            color = np.random.choice(['red', 'green', 'blue', 'purple', 'yellow', 'grey'])
            obj_type = np.random.choice(['key', 'ball', 'box'])
        else:
            color = np.random.choice(['red', 'green', 'blue', 'purple', 'yellow'])
            obj_type = np.random.choice(['box', 'ball'])
        return obj_type, color

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
                task['dropout_goal'] = np.random.uniform() < self.dropout_goal
                task['dropout_correction'] = np.random.uniform() < self.dropout_correction

                # If we have placed all components in place, sanity check that the placement is valid.
                if self.persist_goal and self.persist_agent and self.persist_objs:
                    self.validate_instrs(mission['instrs'])

            except RecursionError as error:
                # self.render(mode="human")
                print('Timeout during mission generation:', error)
                continue

            except RejectSampling as e:
                if tries > 1000:
                    print("ISSUE sampling", e)
                    # self.render(mode='human')
                    raise RejectSampling
                # print("Rejection error", e)
                continue

            break

        return task

    # Functions fo RL2
    def set_task(self, _):
        """
        Sets task dictionary. The parameter is a dummy passed in for compatibility with the normal RL2 set task function
        """
        self.task = self.sample_task()
        self.itr = 0

    # Functions for RL2
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
        for i in range(self.num_rows):
            for j in range(self.num_cols):
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
        self.idx = [x for _,x in sorted(zip(dist_pos_unwrapped, idx))]

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

        def make_door_toggle(obj):
            def toggle(env, pos):
                # If the player has the right key to open the door
                if obj.is_locked:
                    if isinstance(env.carrying, Key) and env.carrying.color == obj.color:
                        obj.is_locked = False
                        obj.is_open = True
                        return True
                    return False

                obj.is_open = True
                return True
            return toggle

        for obj in objs:
            if obj.type == 'box':
                obj.contains = obj
            if obj.type == 'door':
                obj.toggle = make_door_toggle(obj)

    def gen_mission(self):
        """
        Generate the mission for a single meta-task.  Any environment setup elements in the self.task dictionary
        are loaded from there.  All others are randomly sampled. Also decides whether to dropout goal and/or
        corrections for this meta-task.
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

        if 'dropout_goal' in self.task:
            self.dropout_current_goal = self.task['dropout_goal']
        else:
            self.dropout_current_goal = False
        if self.dropout_independently and self.dropout_current_goal:
            self.dropout_current_correction = False
        elif 'dropout_correction' in self.task:
            self.dropout_current_correction = self.task['dropout_correction']
        else:
            self.dropout_current_correction = False


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
        misc = ["follow_teacher"]
        return colors + types + actions + fillers + misc

    def to_vocab_index(self, mission, pad_length=None):  # TODO: turn this into an embedding
        """
        Take a mission string, and return a fixed-length vector where each index is the index of the nth word in the
        mission.  The end is padded with -1
        :param mission: mission text as a string
        :param pad_length: length to pad the mission string to
        :return: list of integer indices of length pad_length (or len(mission.split(" ")) if pad_length is not provided)
        """
        words = mission.split(" ")
        vocab = self.vocab()
        mission_list = [vocab.index(word) for word in words]
        if pad_length is not None:
            mission_list = mission_list + [-1] * (pad_length - len(mission_list))
        if len(mission_list) > pad_length:
            raise ValueError("Mission is too long: " + mission + str(pad_length))
        return mission_list


    def gen_obs(self):
        """
        Generate the agent's view (partially observable). It's a concatenation of the agent's direction and position,
        the flattened partially observable view in front of it, the encoded mission string, and the teacher's feedback,
        if available.
        :return: np array of the agent's observation
        """
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # self.compute_obj_infos()  # TODO: necessary?
        grid_representation = image.flatten()
        goal = self.to_vocab_index(self.mission, pad_length=10)
        if self.dropout_current_goal:
            goal = -np.ones_like(goal)
        obs = np.concatenate([[self.agent_dir],
                               self.agent_pos,
                               grid_representation,
                               goal])
        if hasattr(self, 'teacher') and self.teacher is not None:
            if self.dropout_current_correction:
                correction = self.teacher.empty_feedback()
            else:
                correction = self.teacher.give_feedback([obs])
            obs = np.concatenate([obs, correction])
        return deepcopy(obs)

    def step(self, action):
        """
        Step both the env and the teacher
        :param action: action to take
        :return: results of stepping the env
        """
        obs, rew, done, info = super().step(action)
        info['agent_pos'] = self.agent_pos
        info['agent_dir'] = self.agent_dir
        info['agent_room'] = self.room_from_pos(*self.agent_pos).top
        info['step'] = self.itr
        info['dropout_goal'] = self.dropout_current_goal
        info['dropout_corrections'] = self.dropout_current_correction

        if hasattr(self, "obj_pos"):
            info['goal_room'] = self.room_from_pos(*self.obj_pos).top
            info['goal_pos'] = self.obj_pos
        else:
            info['goal_room'] = (-1, -1)
            info['goal_pos'] = (-1, -1)

        if hasattr(self, 'teacher') and self.teacher is not None:
            teacher_copy = deepcopy(self.teacher)
            try:
                # Even if we use multiple teachers, presumably they all relate to one underlying path.
                # We can log what action is the next one on this path (currently in teacher.next_action).
                info['teacher_action'] = self.teacher.next_action

                self.teacher.step([action])
                # Update the observation with the teacher's new feedback
                obs = self.gen_obs()

            except Exception as e:
                # self.render('human')
                print("ERROR!!!!!", e)
                teacher_copy.step([action])
        return obs, rew, done, info

    def reset(self):
        """
        Reset both the env and the teacher
        :return: observation from the env reset
        """
        obs = super().reset()
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher.reset()
        self.itr += 1
        return obs
