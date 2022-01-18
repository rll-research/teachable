import os
os.makedirs('logs', exist_ok=True)
import ipywidgets as widgets
from importlib import reload 
import joblib
import os
import pathlib
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scripts.train_model import create_policy
from utils.dictlist import DictList

from envs.babyai.utils.buffer import Buffer
from envs.babyai.bot import OBJ_TYPES
from gym_minigrid.minigrid import TILE_PIXELS, Key, Ball, Box, Door, Wall, COLOR_NAMES
import numpy as np
from envs.babyai.utils.obs_preprocessor import make_obs_preprocessor
from utils.utils import set_seed
import time
import pickle as pkl
import uuid
from scripts.train_model import *
from scripts.arguments import ArgumentParser
import sys
import mediapy as media
import pandas as pd

class HumanFeedback:
    def __init__(self, env_type='BabyAI', collect_type='Advice', save_path=None, seed=1):
        class Args:
            def __init__(self):
                if env_type == 'BabyAI':
                    self.feedback_type = 'OSREasy'
                    self.env_type = 'babyai'
                    self.env = 53
                    self.skip = 1
                    self.model = 'saved_models/babyai_offset_advice'
                elif env_type == 'Ant':
                    self.feedback_type = 'OffsetWaypoint'
                    self.env_type = 'ant'
                    self.env = 2
                    self.skip = 10  # TODO: what value???
                    self.model = 'saved_models/ant_offset_advice'
                else:
                    raise NotImplementedError
                self.no_save = save_path is None
                self.advance = 'scroll'
                self.num_trajs = 100000
                self.seed = seed
                if collect_type == 'Precollected':
                    self.save_path = 'logs/precollected_babyai_buffer'
                else:
                    self.save_path = 'logs/' + save_path + '_buffer'
                self.successful_only = False
                self.demos = collect_type == 'Demos'
                self.val_every = 10
                
        # Load model
        self.args = Args()
        self.policy, self.env, _ = self.load_policy(self.args.model)
        self.env._wrapped_env.static_env = True
        # Load env
        self.env.set_level_distribution(self.args.env)
        # Create buffer
        if not self.args.no_save:
            save_path = pathlib.Path(self.args.save_path)
            self.save_path = save_path
            if not save_path.exists():
                save_path.mkdir()
            self.buffer = Buffer(save_path, self.args.num_trajs, val_prob=.1, successful_only=self.args.successful_only)
        if collect_type == 'Precollected':
            return
        # Create window
        self.window = plt.figure(figsize=(10,10))
        self.window.canvas.mpl_connect('key_press_event', self.key_handler)
        self.window.canvas.mpl_connect('button_press_event', self.onclick)
        self.window.canvas.mpl_connect('scroll_event', self.on_scroll)
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
            pass
        self.num_frames = 1
        self.num_correct = 1
        self.ready = False
        self.collecting_val = False

        self.reset()


    def redraw(self, img):
        if self.args.env_type == 'ant':
            img = self.env.render('rgb_array')
        else:
            vis_mask = self.env.oracle[self.args.feedback_type].vis_mask
            img = self.env.render('rgb_array', tile_size=32, full_vis_mask=vis_mask, highlight=False)
        plt.clf()
        plt.imshow(img)
        plt.show()

    def reset(self):
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
        self.obs = self.env.reset()
#         self.decode_feedback(self.obs[self.args.feedback_type], preprocessed=True, tag='orig')
        self.last_feedback = self.obs[self.args.feedback_type] * 0
        #self.clear_feedback()
        title_str = f"Trajectory {self.num_trajs}, frame {self.num_frames}, Last: {self.last}"
        if self.collecting_val:
            title_str += ' (validation)'
        plt.title(title_str)
        if hasattr(self.env, 'mission'):
            plt.title(self.env.mission)
        self.redraw(self.obs)
        self.ready = True

    def load_policy(self, path):
        path = os.path.join(os.getcwd(), path)
        exp_path = os.path.join(path, 'latest.pkl')
        exp_data = joblib.load(exp_path)
        obs_preprocessor = make_obs_preprocessor([self.args.feedback_type])
        env = exp_data['env']
        args = exp_data['args']
        policy = create_policy(path, self.args.feedback_type, env, args, obs_preprocessor)
        set_seed(self.args.seed)
        env.seed(self.args.seed)
        return policy, env, args

    def clear_feedback(self):
        return # TODO: add back if we want to auto-swap-in no teacher
        empty_feedback = self.env.teacher.empty_feedback()
        self.obs.update(empty_feedback)
        self.current_feedback_type = 'none'

    def step(self, action=None, demo=False):
        print("stepping")
        assert False
        if not self.ready:
            return
        for i in range(self.args.skip):
            self.num_frames += 1
            print("why isn't it printing??????????????????????")
            if not demo:
                print("before feedback", self.obs[self.args.feedback_type])
                self.preprocess_obs()
                self.decode_feedback(self.obs[self.args.feedback_type], preprocessed=True, tag="human-p")
                print("after feedback", self.obs[self.args.feedback_type])
            self.feedback_indicator += 1
            if action is None:
                self.policy.eval()
                action, agent_info = self.policy.get_actions([self.obs])
            new_obs, reward, done, info = self.env.step(action)
            self.advice_count.append(1 if self.steps_since_feedback == 0 else 0)
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
#         self.decode_feedback(new_obs[self.args.feedback_type], preprocessed=True, tag=' orig')


        if done:
            self.last = 'success' if info['success'] else 'timed out'
            self.end_trajectory(self.obs)
        else:
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

    def onclick(self, event):
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
                    x = coord_x
                    y = coord_y
                    #if event.button == 1:  # left click, normal waypoint
                    #    x = round(coord_x)
                    #    y = round(coord_y)
                    #elif event.button == 3:  # right click, goal
                    #    x = coord_x
                    #    y = coord_y
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
            self.obs[self.args.feedback_type] = np.concatenate([self.obs[self.args.feedback_type], indicator])

    def decode_feedback(self, feedback, preprocessed=True, tag=''):  # TODO: eventually delete this
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
        return
        self.ready = True
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
            curr_feedback[feedback] = 1
            self.obs[self.args.feedback_type] = curr_feedback
            self.last_feedback = curr_feedback
        else:
            self.obs[self.args.feedback_type] = feedback
        for _ in range(self.advance_count):
            self.step()
#         self.decode_feedback(self.obs[self.args.feedback_type].copy(), preprocessed=False, tag="human-nop")

    def end_trajectory(self, final_obs):
        self.num_trajs += 1
        self.full_done[-1] = 1
        if not self.args.no_save:
            # Save buffer
            env_infos = {
                'advice_count': torch.IntTensor(self.advice_count),
                'success': np.array(self.full_done),
            }

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            traj_dict = {
                'obs': self.obs_list,
                'next_obs': self.obs_list[1:] + [final_obs],
                'action': torch.FloatTensor(np.concatenate(self.action_list)).to(device),
                # 'action_probs': self.action_probs,
                'teacher_action': torch.FloatTensor(self.teacher_action),
                'full_done': torch.FloatTensor(self.full_done),
                'env_infos': DictList(env_infos)
            }
            assert len(traj_dict['teacher_action'].shape) == len(traj_dict['full_done'].shape) == 1
            traj = DictList(traj_dict)
            self.buffer.add_batch(traj, trim=True, only_val=self.num_trajs == self.args.val_every - 1)
            path = self.save_path.joinpath('timesteps.pkl')
            time_dict = {'timesteps': self.timesteps, 'times': self.times}
            with open(path, 'wb') as f:
                pkl.dump(time_dict, f)
        # Reset
        self.reset()

    def key_handler(self, event):
        demo = self.args.demos
        # if event.key == ' ':
        #     self.step()
        #     return
        if event.key == 'v':
            self.collecting_val = not self.collecting_val
        if event.key == 'r':
            self.last = 'manual reset'
            self.end_trajectory(self.obs)
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


def make_args(collector, save_path):
    sys.argv = sys.argv[:1]
    parser = ArgumentParser()
    args = parser.parse_args()
    args.prefix = save_path
    args.env = collector.args.env_type
    args.level = collector.args.env
    args.buffer_path = collector.args.save_path
    if collector.buffer.counts_train == 0:
        raise ValueError("Please collect data before training!")
    args.distill_teacher = 'none'
    args.num_rollouts = 1#5  TODO: fix this!
    args.log_interval = 1
    args.horizon = 60
    args.buffer_capacity = 100000
    args.n_itr = 50
    args.no_tb = True
    return args


def display_trained_model(save_path):
    url = f'logs/{save_path}/vid.avi'
    video = media.read_video(url)
    media.show_video(video)
    

def load_data(name, file_name='progress.csv'):
    csv_name = pathlib.Path.cwd().joinpath('logs', name, file_name)
    data = pd.read_csv(csv_name)
    data.columns = [c.strip() for c in data.columns]
    return data

def plot(run_name, metric='success_rate', x_label='Itrs'):
    use_itrs = x_label in ['Itrs', 'Samples']
    data = load_data(run_name, file_name='results.csv')
    data.columns = ['policy_env','policy','env','success_rate','stoch_accuracy','itr','num_feedback','time','reward']
    y = data[metric].ewm(span=5).mean().to_numpy()
    if use_itrs:
        x = data['itr'].to_numpy()
    else:
        x = data['num_feedback'].to_numpy()
    plt.title(run_name)
    plt.plot(x, y)
    plt.ylabel('Success', fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.show()


