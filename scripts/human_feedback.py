import argparse
import joblib
import os
import pathlib
import torch
import matplotlib
matplotlib.use('TkAgg')
from babyai.utils.buffer import Buffer
from babyai.bot import OBJ_TYPES
from gym_minigrid.window import Window
from gym_minigrid.minigrid import TILE_PIXELS, Key, Ball, Box, Door, Wall, COLOR_NAMES
import numpy as np
import matplotlib.pyplot as plt
from babyai.rl.utils.dictlist import DictList
from babyai.utils.obs_preprocessor import make_obs_preprocessor
from meta_mb.utils.utils import set_seed
import time
import pickle as pkl

# agent_x = 255
# agent_y = 250

# Works for L0
AGENT_X = 248
AGENT_Y = 252
D4RL_TILE_PIXELS = 75 # Works for L0

# # WORKS FOR L2
AGENT_X = 202
AGENT_Y = 296
D4RL_TILE_PIXELS = 49#64 # WORKS for L2

class HumanFeedback:
    def __init__(self):
        self.args = self.make_args()
        # Load model
        self.policy, self.env, _, self.saved_model = self.load_policy(self.args.model)
        if self.args.target_policy is not None:
            target_policy, _, _, _ = self.load_policy(self.args.target_policy)
            self.policy['none'] = target_policy['none']
        # Load env
        self.env.set_level_distribution(self.args.env)
        # Create buffer
        save_path = pathlib.Path(self.args.save_path)
        self.save_path = save_path
        if not save_path.exists():
            save_path.mkdir()
        self.buffer = Buffer(save_path, self.args.num_trajs, 1, val_prob=.1, successful_only=self.args.successful_only)
        self.teacher_null_dict = self.env.teacher.null_feedback()
        self.teacher_dict = {k: k == self.args.feedback_type for k in self.teacher_null_dict.keys()}
        self.obs_preprocessor = make_obs_preprocessor(self.teacher_null_dict, include_zeros=False)
        # Create window
        self.window = Window('gym_minigrid - ' + str(self.args.env))
        self.window.reg_key_handler(self.key_handler)
        self.window.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.window.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.num_trajs = 0
        self.obs = None
        self.obs_list = []
        self.action_list = []
        self.action_probs = []
        self.teacher_action = []
        self.full_done = []
        self.advice_count = []
        self.advance_count = 0
        self.current_feedback_type = self.args.feedback_type
        self.feedback_indicator = 0
        self.steps_since_feedback = 0
        self.last = None
        try:
            self.advance_count = int(self.args.advance)
        except:
            print(self.args.advance)
        self.num_frames = 1
        self.num_correct = 1
        self.ready = False
        self.collecting_val = False

        self.reset()

        # Blocking event loop
        self.window.show(block=True)

    def redraw(self, img):
        if not self.args.agent_view:
            try:
                vis_mask = self.env.oracle[self.args.feedback_type].vis_mask
                # img = self.env.render('rgb_array')
                img = self.env.render('rgb_array', tile_size=self.args.tile_size, full_vis_mask=vis_mask, highlight=False)
            except:
                img = self.env.render('rgb_array')
        self.window.show_img(img)

    def reset(self):
        self.ready = False
        # If we're training concurrently, reload so we get the new model
        if self.args.train_concurrently:
            self.policy, _, _, self.saved_model = self.load_policy(self.args.model)
            if self.args.target_policy is not None:
                target_policy, _, _, _ = self.load_policy(self.args.target_policy)
                self.policy['none'] = target_policy['none']
        for agent in self.policy.values():
            agent.reset(dones=[True])
        self.obs_list = []
        self.action_list = []
        self.action_probs = []
        self.teacher_action = []
        self.full_done = []
        self.advice_count = []
        self.times = []
        self.timesteps = []
        self.timestep_counter = 0
        self.start_time = time.time()
        self.env.set_task()
        print("=" * 100)
        self.obs = self.env.reset()
        self.decode_feedback(self.obs[self.args.feedback_type], preprocessed=True, tag='orig')
        self.last_feedback = self.obs[self.args.feedback_type] * 0
        self.clear_feedback()
        title_str = f"Trajectory {self.num_trajs}, frame {self.num_frames}, Last: {self.last}"
        if self.collecting_val:
            title_str += ' (validation)'
        plt.title(title_str)
        # plt.title(f"Trajectory {self.num_trajs}, frame {self.num_frames}, acc {self.num_correct / self.num_frames}")
        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)
            self.window.set_caption(self.env.mission)
        # print("TEACHER ACTION:", self.env.teacher_action)
        self.redraw(self.obs)


    def load_policy(self, path):
        base_path = os.path.join(os.getcwd(), "data")
        path = os.path.join(base_path, path)
        print("PATH", path)
        saved_model = joblib.load(path)
        env = saved_model['env']
        set_seed(self.args.seed)
        env.seed(self.args.seed)
        policy = saved_model['policy']
        args = saved_model['args']
        # for p_dict in policy.values():
        #     p_dict.instr_rnn.flatten_parameters()
        return policy, env, args, saved_model

    def clear_feedback(self):
        return # TODO: add back if we want to auto-swap-in no teacher
        empty_feedback = self.env.teacher.empty_feedback()
        self.obs.update(empty_feedback)
        self.current_feedback_type = 'none'

    def make_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--env",
            help="level to load",
            type=int,
            default=25
        )
        parser.add_argument(
            "--model",
            type=str,
            required=True
        )
        parser.add_argument(
            "--feedback_type",
            type=str,
            default="PreActionAdvice"
        )
        parser.add_argument(
            '--advance',
            default='scroll',
        )
        parser.add_argument(
            '--speed',
            default=0.1,
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=0,
        )
        parser.add_argument(
            '--successful_only',
            action='store_true'
        )
        parser.add_argument(
            '--no_save',
            action='store_true'
        )
        parser.add_argument(
            '--save_path',
            type=str,
            default='TEMP'
        )
        parser.add_argument(
            "--tile_size",
            type=int,
            help="size at which to render tiles",
            default=32
        )
        parser.add_argument(
            "--skip",
            type=int,
            help="number of times to repeat acctions",
            default=1
        )
        parser.add_argument(
            '--agent_view',
            default=False,
            help="draw the agent sees (partially observable view)",
            action='store_true'
        )
        parser.add_argument(
            '--demos',
            action='store_true'
        )
        parser.add_argument(
            '--target_policy',
            default=None,
        )
        parser.add_argument(
            '--train_concurrently',
            action='store_true'
        )
        parser.add_argument(
            '--verbose',
            action='store_true'
        )

        parser.add_argument(
            '--env_type',
            default='babyai'
        )
        parser.add_argument(
            '--num_trajs',
            type=int,
            default=100000,
        )
        args = parser.parse_args()
        return args

    def step(self, action=None, demo=False):
        if not self.ready:
            return
        for i in range(self.args.skip):
            self.num_frames += 1
            if not demo:
                self.preprocess_obs()
                self.decode_feedback(self.obs[self.args.feedback_type], preprocessed=True, tag="human-p")
            self.feedback_indicator += 1  # TODO: redundant with the other indicator
            if action is None:
                teacher_dict = {k: k == self.current_feedback_type for k in self.teacher_null_dict.keys()}
                o = self.obs_preprocessor([self.obs], teacher_dict, show_instrs=False)  # TODO: show instrs flag
                agent = self.policy[self.current_feedback_type]
                agent.eval()
                action, agent_info = agent.get_actions_t(o, temp=1)
            teacher_action = self.env.teacher_action
            actions = self.env._wrapped_env.Actions
            if self.args.env_type == 'babyai' and action.item() in [actions.pickup, actions.drop]:
                self.ready = False
            new_obs, reward, done, info = self.env.step(action)
            if self.args.env_type == 'babyai' and self.args.feedback_type == 'OSREasy':
                # If we've reached the subgoal, wait
                if np.array_equal(self.env.agent_pos, self.last_feedback[1:3]):
                    self.ready = False
                self.advice_count.append(1 if self.steps_since_feedback == 0 else 0)
            else:
                print("Currently at", self.env.agent_pos, "going to", self.last_feedback[1:3])
            self.steps_since_feedback += 1
            self.timestep_counter += 1
            self.obs_list.append(self.obs)
            self.action_list.append(action)
            self.teacher_action.append(0)
            self.full_done.append(done)
            self.times.append(time.time() - self.start_time)
            self.timesteps.append(self.timestep_counter)
            self.obs = new_obs
            action = None

            if done:
                break
        self.decode_feedback(new_obs[self.args.feedback_type], preprocessed=True, tag=' orig')


        if done:
            self.last = 'success' if info['success'] else 'timed out'
            self.end_trajectory()
        else:
            # if np.random.uniform() < .5:
            self.redraw(self.obs)

    def preprocess_obs(self):
        if not self.args.feedback_type == 'PreActionAdvice':
            self.set_feedback()
        feedback_obs = self.obs[self.args.feedback_type]
        if self.args.feedback_type == 'OffsetWaypoint':
            feedback_obs[:] -= self.env.get_pos()
        if self.args.feedback_type in ['SubgoalCorrections', 'SubgoalSimple']:
            # Add agent pos and dir
            feedback_obs[-1] = self.env.agent_dir / 3
            feedback_obs[-3: -1] = (self.env.agent_pos - 12) / 12
            # Change target to offset
            feedback_obs[-5: -3] = (feedback_obs[-5: -3] - self.env.agent_pos) / 10
        elif self.args.feedback_type in ['OFFIO', 'OFFSparse', 'OFFSparseRandom', 'OSRPeriodicImplicit', 'OSREasy']:
            # Add feedback indicator
            feedback_freq = self.env.teacher.teachers[self.args.feedback_type].feedback_frequency  # TODO: add SSF
            feedback_indicator = np.zeros(feedback_freq)
            steps_since_feedback = min(self.steps_since_feedback, feedback_freq - 1)
            feedback_indicator[steps_since_feedback] = 1
            feedback_obs[-feedback_freq:] = feedback_indicator
            # Turn the absolute coordinate into an offset
            if steps_since_feedback == 0 or self.args.feedback_type in ['OSREasy', 'OSRPeriodicImplicit']:
                feedback_obs[1:3] = feedback_obs[1:3] - self.env.agent_pos
            # Add agent pos and dir
            if steps_since_feedback == 0 or self.args.feedback_type in ['OSREasy', 'OSRPeriodicImplicit', 'OFFSparseRandom']:
                feedback_obs[-feedback_freq - 1] = self.env.agent_dir / 3
                feedback_obs[-feedback_freq - 3: -feedback_freq - 1] = (self.env.agent_pos - 12) / 12
            else:
                feedback_obs[-feedback_freq - 1] = -1
                feedback_obs[-feedback_freq - 3: -feedback_freq - 1] = np.array([-1, -1])

    def postprocess_obs(self):
        # Return if we aren't tracking count
        return # TODO: re-add this if we want to auto-swap-in no teacher
        # if self.args.feedback_type in ['SubgoalCorrections', 'SubgoalSimple']:
        #     return
        # feedback_obs = getattr(self.obs, self.args.feedback_type)
        # if self.args.feedback_type in ['OFFIO', 'OFFSparse', 'OFFSparseRandom']:

    def onclick(self, event):
        self.ready = True
        try:
            ix, iy = event.xdata, event.ydata
            pixels = TILE_PIXELS if self.args.env_type == 'babyai' else D4RL_TILE_PIXELS
            coord_width = ix / pixels
            coord_height = iy / pixels

            if self.args.env_type == 'd4rl':
                coord_x = (ix - AGENT_X) / D4RL_TILE_PIXELS
                coord_y = (iy - AGENT_Y) / D4RL_TILE_PIXELS
                if self.args.feedback_type == 'Direction':
                    dir = np.array([coord_x, coord_y])
                    self.set_feedback(dir)
                    return
                elif self.args.feedback_type in ['Waypoint', 'OffsetWaypoint']:
                    if event.button == 1:  # left click, normal waypoint
                        x = round(coord_x)
                        y = round(coord_y)
                    elif event.button == 3:  # right click, goal
                        x = coord_x
                        y = coord_y
                    self.set_feedback(np.array([-y, x], dtype=np.float64))
                    return

            x = int(coord_width)
            y = int(coord_height)
            offset_x = x - self.env.agent_pos[0]
            offset_y = y - self.env.agent_pos[1]
            # choose the direction based on which side of the cell we're closest to
            left_diff = coord_width - x
            right_diff = x + 1 - coord_width
            top_diff = coord_height - y
            bottom_diff = y + 1 - coord_height
            agent_dir = np.argmin([top_diff, right_diff, bottom_diff, left_diff])
            agent_diff = agent_dir - self.env.agent_dir

            if self.args.feedback_type == 'OFFIO':
                coords = np.zeros(4)
                coords[0] = offset_x
                coords[1] = offset_y
                if agent_diff == 3:
                    agent_diff = -1
                elif agent_diff == -3:
                    agent_diff = 1
                coords[2] = agent_diff
                coords[3] = self.env.agent_dir
                self.set_feedback(coords)
            elif self.args.feedback_type in ['OSRPeriodicImplicit', 'OSREasy']:
                is_obj = 1 if type(self.env.grid.get(x, y)) in [Key, Ball, Box, Door] else 0
                coords = np.zeros(3 + 3 + self.env.teacher.teachers[self.args.feedback_type].feedback_frequency)
                coords[0] = is_obj
                coords[1] = x
                coords[2] = y
                self.set_feedback(coords)
            elif self.args.feedback_type in ['OFFSparse', 'OFFSparseRandom']:
                is_obj = 1 if type(self.env.grid.get(x, y)) in [Key, Ball, Box, Door] else 0
                coords = np.zeros(3 + 3 + self.env.teacher.teachers[self.args.feedback_type].feedback_frequency)
                coords[0] = is_obj
                coords[1] = offset_x
                coords[2] = offset_y
                # self.decode_offset(coords, preprocessed=False)
                self.set_feedback(coords)
            elif self.args.feedback_type in ['SubgoalCorrections', 'SubgoalSimple']:
                obj = self.env.grid.get(x, y)
                subgoal_names = ['OpenSubgoal',
                                 'DropSubgoal',
                                 'PickupSubgoal',
                                 'GoNextToSubgoal']
                subgoal_idx_all = np.zeros(len(subgoal_names) + 1
                                           + len(COLOR_NAMES) + 1
                                           + len(OBJ_TYPES) + 1
                                           + 2
                                           + 3)
                if type(obj) is Door:
                    if obj.is_open or event.button == 1:
                        subgoal_name = 'GoNextToSubgoal'
                    else:
                        subgoal_name = 'OpenSubgoal'
                elif type(obj) in [Key, Box, Ball]:
                    if event.button == 1:  # left click, GoTo
                        subgoal_name = 'GoNextToSubgoal'
                    elif self.env.carrying:
                        subgoal_name = 'DropSubgoal'
                    else:
                        subgoal_name = 'PickupSubgoal'
                elif obj is None or type(obj) == Wall:
                    if event.button == 1:  # left click, GoTo
                        subgoal_name = 'GoNextToSubgoal'
                    elif event.button == 3:  # right click, PickUp
                        subgoal_name = 'DropSubgoal'
                    else:
                        print("huh2?", event.button)
                else:
                    print(x, y)
                    print(f"OBJ is {type(obj)}; invalid subgoal")
                    print(f"OBJ is {obj is None}; invalid subgoal")
                    return
                subgoal_val = np.array([x, y])
                if type(obj) in [Box, Ball, Key, Door]:
                    color_idx = COLOR_NAMES.index(obj.color)
                    type_idx = OBJ_TYPES.index(obj.type)
                else:
                    color_idx = len(COLOR_NAMES)
                    type_idx = len(OBJ_TYPES)

                # Index the subgoal type
                subgoal_idx_all[subgoal_names.index(subgoal_name)] = 1.0
                curr_idx = len(subgoal_names) + 1
                # Index target object color
                subgoal_idx_all[curr_idx + color_idx] = 1.0
                curr_idx += len(COLOR_NAMES) + 1
                # Index target object name
                subgoal_idx_all[curr_idx + type_idx] = 1.0
                curr_idx += len(OBJ_TYPES) + 1
                # Index the target coordinate
                subgoal_idx_all[curr_idx:curr_idx + 2] = subgoal_val
                curr_idx += 2
                # Index current agent position
                subgoal_idx_all[curr_idx: curr_idx + 2] = (self.env.agent_pos - 12) / 12
                curr_idx += 2
                # Index current agent orientation
                subgoal_idx_all[curr_idx] = self.env.agent_dir / 3
                self.set_feedback(subgoal_idx_all)


        except Exception as e:
            print("invalid coordinate", e)

    def add_feedback_indicator(self):
        if self.args.feedback_type in ['OFFIO', 'OFFSparseRandom', 'OSRPeriodicImplicit', 'OFFSR', 'OSREasy']:
            indicator = self.env.teacher.teachers[self.args.feedback_type].get_last_feedback_indicator()
            print("INDICATOR", indicator)
            self.obs[self.args.feedback_type] = np.concatenate([self.obs[self.args.feedback_type], indicator])

    def decode_feedback(self, feedback, preprocessed=True, tag=''):
        if not self.args.verbose:
            return
        if self.args.feedback_type == 'Direction':
            self.decode_direction(feedback.copy(), preprocessed, tag)
        elif self.args.feedback_type == 'Cardinal':
            self.decode_cardinal(feedback.copy(), preprocessed, tag)
        elif self.args.feedback_type  == 'Waypoint':
            self.decode_waypoint(feedback.copy(), preprocessed, tag)
        elif self.args.feedback_type == 'OffsetWaypoint':
            self.decode_offsetwaypoint(feedback.copy(), preprocessed, tag)
        if self.args.feedback_type in ['OFFIO', 'OFFSparseRandom', 'OSRPeriodicImplicit', 'OFFSR', 'OSREasy']:
            self.decode_offset(feedback.copy(), preprocessed, tag)
        elif self.args.feedback_type in ['SubgoalCorrections', 'SubgoalSimple']:
            self.decode_subgoal(feedback.copy(), preprocessed, tag)

    def decode_direction(self, feedback, _, tag):
        print(f"{tag} Head in direction {feedback}")

    def decode_cardinal(self, feedback, _, tag):
        index = np.argmax(feedback)
        dir = ['left', 'up', 'right', 'down'][index]
        print(f"{tag} Head in direction {dir}")

    def decode_offsetwaypoint(self, feedback, preprocessed, tag):
        if not preprocessed:
            # TODO: okay??
            og = feedback.copy()
            feedback = feedback - self.env.get_pos()
        og = None
        print(f"{tag} OffsetWaypoint: {feedback}, {og}")

    def decode_waypoint(self, feedback, _, tag):
        print(f"{tag} Waypoint: {feedback * 15}")

    def decode_offset(self, offset, preprocessed=True, tag=""):  # TODO: currently only handles sparse
        first = offset[0]
        coords_offset = offset[1:3]
        start_str = "Using an obj at " if first else "Going to"
        if preprocessed:
            agent_pos = offset[3: 5] * 12 + 12
            agent_dir = offset[5] * 3
            if agent_dir < 0:
                agent_dir = offset[5]
                agent_pos = offset[3: 5]
            timesteps_ago = np.argmax(offset[6:])
        else:
            agent_pos = agent_dir = timesteps_ago = -1

            coords_offset -= self.env.agent_pos

        print(f"{tag} {start_str} {coords_offset}, {timesteps_ago} timesteps ago"
              f" pos {agent_pos}, dir {agent_dir}")

    def decode_subgoal(self, subgoal, preprocessed=True, tag=''):
        # Subgoal Name
        subgoal_names = ['OpenSubgoal',
                         'DropSubgoal',
                         'PickupSubgoal',
                         'GoNextToSubgoal']
        subgoal_name = subgoal_names[np.argmax(subgoal[:len(subgoal_names)]).item()]
        curr_idx = len(subgoal_names) + 1
        # Obj color
        obj_color = (COLOR_NAMES + ['none'])[np.argmax(subgoal[curr_idx: curr_idx + len(COLOR_NAMES) + 1]).item()]
        curr_idx += len(COLOR_NAMES) + 1
        # Obj name
        obj_type = (OBJ_TYPES + ['none'])[np.argmax(subgoal[curr_idx: curr_idx + len(OBJ_TYPES) + 1]).item()]
        curr_idx += len(OBJ_TYPES) + 1
        # Target coordinate
        coordinate = subgoal[curr_idx: curr_idx + 2]
        curr_idx += 2
        # Agent pos
        agent_pos = subgoal[curr_idx: curr_idx + 2] * 12 + 12
        curr_idx += 2
        # Agent Dir
        agent_dir = subgoal[curr_idx] * 3
        if preprocessed:
            coordinate = (coordinate * 10) + agent_pos
        print(f"{tag} Name: {subgoal_name}, Coord: {coordinate}, "
               f"obj {obj_color} {obj_type}, pos {agent_pos}, dir {agent_dir}")

    def on_scroll(self, event):
        if self.args.advance == 'scroll':
            self.step()

    def set_feedback(self, feedback=None, demo=False):
        if self.args.demos and demo:
            action = np.array([int(feedback)])
            self.step(action, demo)
            return
        if feedback is None:
            feedback = self.last_feedback
        else:
            self.steps_since_feedback = 0
            self.obs['gave_' + self.args.feedback_type] = 1.0
        self.last_feedback = feedback.copy()
        self.feedback_indicator = 0
        self.current_feedback_type = self.args.feedback_type

        if self.args.feedback_type == 'PreActionAdvice':
            feedback = int(feedback)
            assert feedback >= 0
            assert feedback <= 7
            curr_feedback = np.zeros(8)
            # print("argmax", feedback)
            curr_feedback[feedback] = 1
            self.obs[self.args.feedback_type] = curr_feedback
            self.last_feedback = curr_feedback
        else:
            self.obs[self.args.feedback_type] = feedback
        for _ in range(self.advance_count):
            self.step()
        self.decode_feedback(self.obs[self.args.feedback_type].copy(), preprocessed=False, tag="human-nop")

    def end_trajectory(self):
        self.num_trajs += 1
        self.full_done[-1] = 1
        if not self.args.no_save:
            # Save buffer
            env_infos = {
                'advice_count': torch.IntTensor(self.advice_count),
                'success': torch.FloatTensor(self.full_done),
            }

            traj_dict = {
                'obs': self.obs_list,
                'action': torch.FloatTensor(np.concatenate(self.action_list)).cuda(),
                # 'action_probs': self.action_probs,
                'teacher_action': torch.FloatTensor(self.teacher_action),
                'full_done': torch.FloatTensor(self.full_done),
                'env_infos': DictList(env_infos)
            }
            assert len(traj_dict['teacher_action'].shape) == len(traj_dict['full_done'].shape) == 1
            traj = DictList(traj_dict)
            self.buffer.add_batch(traj, int(self.args.env), trim=True, only_val=self.collecting_val)
            path = self.save_path.joinpath('timesteps.pkl')
            time_dict = {'timesteps': self.timesteps, 'times': self.times}
            with open(path, 'wb') as f:
                pkl.dump(time_dict, f)
        # Reset
        self.reset()

    def key_handler(self, event):
        self.ready = True
        demo = self.args.demos
        # if event.key == ' ':
        #     self.step()
        #     return
        if event.key == 'v':
            self.collecting_val = not self.collecting_val
        if event.key == 'r':
            self.last = 'manual reset'
            self.end_trajectory()
            return
        if event.key == 'c':
            self.step()
        if self.args.feedback_type == 'Cardinal':
            arr = np.zeros(4)
            if event.key == 'left':
                arr[0] = 1
            if event.key == 'right':
                arr[2] = 1
            if event.key == 'up':
                arr[1] = 1
            if event.key == 'down':
                arr[3] = 1
            self.set_feedback(arr, demo=False)
            return
        elif self.args.env_type == 'babyai' and (self.args.feedback_type == 'PreActionAdvice' or self.args.demos):
            actions = self.env._wrapped_env.Actions
            if event.key == 'left':
                self.set_feedback(actions.left, demo=demo)
                return
            if event.key == 'right':
                self.set_feedback(actions.right, demo=demo)
                return
            if event.key == 'up':
                self.set_feedback(actions.forward, demo=demo)
                return

            # Spacebar
            if event.key == ' ':
                self.set_feedback(actions.toggle, demo=demo)
                return
            if event.key == 'pageup':
                self.set_feedback(actions.pickup, demo=demo)
                return
            if event.key == 'pagedown':
                self.set_feedback(actions.drop, demo=demo)
                return
        else:
            raise print("Invalid key", event.key)
        print('pressed', event.key)


HumanFeedback()

# Add ability to load 3 models
# Try training a run with subgoals (collect first, then train)
# Try training a run with PA (collect first, then train)
# Try training a run with offset (collect first, then train)
# Try training a run with mixed curriculum x2
# Get subgoal offset to be 10x
# Improve sample efficiency

# ---------

# Undo button?
# Add option to train midway
# Have running list of pasts feedback/actions
# set up asynchronous training, train continuously x3
# GOAL: have this done by Mon night!

# CURRENT ISSUE
# - Trying to get coordinate offsets, but tapping the bottom-right corner doesn't coreespond to the image width/height
