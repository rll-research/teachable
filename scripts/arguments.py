"""
Common arguments for BabyAI training scripts
"""

import os
import argparse
import numpy as np


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Env arguments
        self.add_argument("--env", default='babyai',
                            help="name of the environment to train on")
        self.add_argument('--level', type=int, default=0)
        self.add_argument('--env_dist', type=str, default='one_hot',
                          choices=["one_hot", "smooth", "uniform", 'four_levels', 'four_big_levels', 'five_levels',
                                   'goto_levels', 'easy_goto'])
        self.add_argument('--fully_observed', action='store_true')
        self.add_argument('--reward_type', type=str, choices=['dense', 'sparse', 'oracle_action', 'oracle_dist',
                                                              'vector_dir', 'vector_dir2', 'vector_dir_final',
                                                              'vector_dir_waypoint', 'vector_dir_both',
                                                              'vector_dir_waypoint_negative', 'waypoint',
                                                              'vector_next_waypoint', 'wall_penalty', 'dense_pos_neg',
                                                              'dense_success'],
                          default='default_reward')
        self.add_argument('--leave_out_object', action='store_true')
        self.add_argument('--static_env', action='store_true')
        self.add_argument('--eval_envs', nargs='+', type=int, default=None)
        self.add_argument('--horizon', type=str, default='default')

        # Training arguments
        self.add_argument("--seed", type=int, default=1,
                          help="random seed; if -1, a random random seed will be used  (default: 1)")
        self.add_argument("--epochs", type=int, default=20)
        self.add_argument("--frames_per_proc", type=int, default=40,
                          help="number of frames per process before update (default: 40)")
        self.add_argument("--beta1", type=float, default=0.9,
                          help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                          help="beta2 for Adam (default: 0.999)")
        self.add_argument("--optim_eps", type=float, default=1e-5,
                          help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
        self.add_argument("--optim_alpha", type=float, default=0.99,
                          help="RMSprop optimizer apha (default: 0.99)")
        self.add_argument("--batch_size", type=int, default=512,
                          help="batch size for distillation")
        self.add_argument("--entropy_coef", type=float, default=0.001,
                          help="entropy term coefficient")
        self.add_argument('--distill_entropy_coef', type=float, default=0)
        self.add_argument('--n_itr', type=int, default=100000)
        self.add_argument('--end_on_full_buffer', action='store_true')
        self.add_argument('--algo', type=str, default='ppo', choices=['sac', 'ppo', 'hppo'])
        self.add_argument('--min_itr_steps', type=int, default=0)
        self.add_argument('--min_itr_steps_distill', type=int, default=0)
        self.add_argument('--lr', type=float, default=1e-3)
        self.add_argument('--discount', type=str, default='default')
        self.add_argument('--gae_lambda', type=float, default=.95)
        self.add_argument('--num_envs', type=int, default=5)
        self.add_argument('--early_stop', type=int, default=float('inf'))
        self.add_argument('--early_stop_metric', type=str, default=None)
        self.add_argument('--no_buffer', action='store_true')
        self.add_argument('--num_rollouts', type=int, default=5)
        self.add_argument('--hide_instrs', action='store_true')
        self.add_argument('--padding', action='store_true')
        self.add_argument('--feedback_from_buffer', action='store_true')
        self.add_argument('--hidden_dim', type=int, default=128)
        self.add_argument('--instr_dim', type=int, default=128)
        self.add_argument('--sequential', action='store_true')
        self.add_argument('--clip_eps', type=float, default=.2)

        # Saving/loading/logging
        self.add_argument('--prefix', type=str, default='DEBUG')
        self.add_argument('--save_option', type=str, default='level',
                          choices=['all', 'level', 'latest', 'none', 'gap'])
        self.add_argument('--save_untrained', action='store_true')
        self.add_argument("--log_interval", type=int, default=20)
        self.add_argument("--eval_interval", type=int, default=20)
        self.add_argument('--no_video', action='store_true')

        # Teacher
        self.add_argument('--feedback_freq', nargs='+', type=int, default=[1])
        self.add_argument('--collect_with_oracle', action='store_true')
        self.add_argument('--reload_exp_path', type=str, default=None)
        self.add_argument('--continue_train', action='store_true')

        # Policies
        self.add_argument('--collect_policy', default=None, help='path to collection policy')
        self.add_argument('--collect_teacher', default=None,
                          help="Teacher to for collection. If None, no collection happens. If no teacher, put 'none'")
        self.add_argument('--rl_policy', default=None, help='path to rl policy')
        self.add_argument('--rl_teacher', default=None)
        self.add_argument('--distill_policy', default=None, help='path to distill policy')
        self.add_argument('--distill_teacher', default=None)
        self.add_argument('--collect_with_rl_policy', action='store_true')
        self.add_argument('--collect_with_distill_policy', action='store_true')
        self.add_argument('--relabel_policy', default=None, help='path to relabel policy')
        self.add_argument('--relabel_teacher', default=None)

        # Distillations
        self.add_argument('--distillation_steps', type=int, default=15)
        self.add_argument('--buffer_capacity', type=int, default=1)
        self.add_argument('--buffer_path', type=str, default=None)
        self.add_argument('--distill_dropout_prob', type=float, default=0.)
        self.add_argument('--collect_dropout_prob', type=float, default=0.)
        self.add_argument('--distill_successful_only', action='store_true')
        self.add_argument('--kl_coef', type=float, default=0)
        self.add_argument('--control_penalty', type=float, default=.01)
        self.add_argument('--recon_coef', type=float, default=0)
        self.add_argument('--z_dim', type=int, default=32)
        self.add_argument('--info_bot', action='store_true')
        self.add_argument('--source', type=str, default='agent', choices=['agent', 'teacher', 'agent_argmax',
                                                                                'agent_probs'])
        self.add_argument('--no_distill', action='store_true')
        self.add_argument('--train_level', action='store_true')

    def parse_args(self, arg=None):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args(arg)

        # Set seed for all randomness sources
        if args.seed == -1:
            args.seed = np.random.randint(10000)

        return args
