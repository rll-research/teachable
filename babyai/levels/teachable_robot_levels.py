from .verifier import *
from babyai.levels.levelgen import RoomGridLevel, RejectSampling
from meta_mb.meta_envs.base import MetaEnv
from gym_minigrid.minigrid import MiniGridEnv, Key, Ball, Box
from gym_minigrid.roomgrid import RoomGrid
import numpy as np
from copy import deepcopy

class Level_TeachableRobot(RoomGridLevel, MetaEnv):
    """
    Go to an object, inside a single room with no doors, some distractors
    """

    def __init__(self, start_loc='all', num_dists=0,
                 include_holdout_obj=True,
                 persist_agent=True, persist_goal=True, persist_objs=True,
                 dropout_goal=0, dropout_correction=0, **kwargs):
        """
        :param start_loc: which part of the grid to start the agent in.  ['top', 'bottom', 'all']
        """
        assert start_loc in ['top', 'bottom', 'all']
        self.start_loc = start_loc
        self.num_dists = num_dists
        self.include_holdout_obj = include_holdout_obj
        self.persist_agent = persist_agent
        self.persist_goal = persist_goal
        self.persist_objs = persist_objs
        self.dropout_goal = dropout_goal
        self.dropout_correction = dropout_correction
        self.task = {}
        # Number of distractors
        super().__init__(**kwargs)

    # Define what starting position to use (train set or hold-out set).  Currently ['top', 'bottom', 'all']
    def set_start_loc(self, start_loc):
        self.start_loc = start_loc

    def resample_task(self):
        self.task = self.sample_tasks(1)[0]

    def sample_object(self):
        if self.include_holdout_obj:
            color = np.random.choice(['red', 'green', 'blue', 'purple', 'yellow', 'grey'])
            obj_type = np.random.choice(['key', 'ball', 'box'])
        else:
            color = np.random.choice(['red', 'green', 'blue', 'purple', 'yellow'])
            obj_type = np.random.choice(['box', 'ball'])
        return obj_type, color

    def make_mission(self):
        raise NotImplementedError


    def sample_task(self):
        while True:
            try:
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
                task['dropout_correction'] = np.random.uniform() < self.dropout_correction  # TODO: consider making this per-timestep

                # If we have placed all components in place, sanity check that the placement is valid.
                if self.persist_goal and self.persist_agent and self.persist_objs:
                    self.validate_instrs(mission['instrs'])

            except RecursionError as error:
                self.render(mode="human")
                self.validate_instrs(mission['instrs'])
                print('Timeout during mission generation:', error)
                continue

            except RejectSampling as e:
                print("Rejection error", e)
                continue

            break

        return task

    # Functions fo RL2
    def set_task(self, _):
        """
        Sets task dictionary.
        """
        self.task = self.sample_task()
        return

    # Functions for RL2
    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.task

    def get_doors(self):
        doors = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)
        return doors

    def get_color_idx(self, color):
        return ['red', 'green', 'blue', 'purple', 'yellow', 'grey'].index(color)

    def get_type_idx(self, obj_type):
        return ['door', 'key', 'ball', 'box', 'lava'].index(obj_type)  # TODO: no lava

    # Convert object positions, types, colors into a vector. Always order by the object positions to stay consistent and so it can't actually memorize.
    def compute_obj_infos(self):
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
        cutoff = int(self.room_size / 2)
        top_index = (0, cutoff) if self.start_loc == 'bottom' else (0, 0)
        bottom_index = (self.room_size, cutoff) if self.start_loc == 'top' else (self.room_size, self.room_size)
        self.place_agent(top_index=top_index, bottom_index=bottom_index)

    def place_in_grid(self, objs):
        for obj in objs:
            i, j = obj.cur_pos
            try:
                self.put_obj(obj, i, j)
            except Exception as e:
                print(e)

    # Generate a mission. Task remains fixed so that object is always spawned and is the goal, but other things can change every time we do a reset.
    def gen_mission(self):

        # First load anything provided in the task
        if 'agent' in self.task:
            self.agent_pos = self.task['agent']['agent_pos']
            self.agent_dir = self.task['agent']['agent_dir']
        if 'mission' in self.task:
            mission = self.task['mission']
        if 'objs' in self.task:
            objs, goal_objs = self.task['objs']
            self.objs = deepcopy(objs)
            self.goal_objs = deepcopy(goal_objs)
            self.place_in_grid(self.objs)

        # Now randomly sample any required information not in the task
        if not 'agent' in self.task:
            self.add_agent()
        if not 'mission' in self.task:
            mission = self.make_mission()
        if not 'objs' in self.task:
            objs, goal_objs = self.add_objs(mission["task"])
            self.objs = deepcopy(objs)
            self.goal_objs = deepcopy(goal_objs)

        self.compute_obj_infos()
        self.instrs = mission['instrs']
        if goal_objs is not None:
            # Currently if there are multiple goal objects, we just pick the first one. # TODO: handle multiple goal objs
            if isinstance(goal_objs, tuple):
                goal_obj = goal_objs[0]
            else:
                goal_obj = goal_objs
            self.obj_pos = goal_obj.cur_pos

        if 'dropout_goal' in self.task:
            self.dropout_current_goal = self.task['dropout_goal']
        else:
            self.dropout_current_goal = False
        if 'dropout_correction' in self.task:
            self.dropout_current_correction = self.task['dropout_correction']
        else:
            self.dropout_current_correction = False


    def place_agent(self, i=None, j=None, rand_dir=True, top_index=None, bottom_index=None):
        """
        Place the agent in a room
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
        colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
        types = ['door', 'key', 'ball', 'box']
        actions = ["go", "pick", "up", "open", "put"]
        fillers = ["to", "next", "the", "a"]
        return colors + types + actions + fillers

    def to_vocab_index(self, mission, pad_length=None):  # TODO: turn this into an embedding
        words = mission.split(" ")
        vocab = self.vocab()
        mission_list = [vocab.index(word) for word in words]
        if pad_length is not None:
            mission_list = mission_list + [0] * (pad_length - len(mission_list))
        if len(mission_list) > pad_length:
            raise ValueError("Mission is too long: " + mission + str(pad_length))
        return mission_list


    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)

        if hasattr(self, 'obj_pos'):
            obj_pos = self.obj_pos
        else:
            obj_pos = np.array([0, 0])

        obs_dict = {
            'direction': self.agent_dir,
            'mission': self.mission,
            'agent_pos': self.agent_pos,
            'obj_pos': obj_pos
        }
        # Compute the object infos
        self.compute_obj_infos()
        grid_representation = image.flatten()
        goal = self.to_vocab_index(self.mission, pad_length=10)
        if self.dropout_current_goal:
            goal = -np.ones_like(goal)
        obs = np.concatenate([[obs_dict['direction']],
                               obs_dict['agent_pos'],
                               grid_representation,
                               goal])
        if hasattr(self, 'teacher') and self.teacher is not None:
            if self.dropout_current_correction:
                correction = self.teacher.give_feedback([obs])[0]
            else:
                correction = self.teacher.empty_feedback()[0]
            obs = np.concatenate([obs, correction])

        return deepcopy(obs)

    def step(self, action):
        results = super().step(action)
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher.step([action])
        return results

    def reset(self):
        obs = super().reset()
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher.reset()
        return obs
