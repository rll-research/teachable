"""
Common arguments for BabyAI training scripts
"""

import os
import argparse
import numpy as np


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--env", default='babyai',
                            help="name of the environment to train on")
        self.add_argument("--model", default=None,
                          help="name of the model (default: ENV_ALGO_TIME)")
        self.add_argument("--pretrained-model", default=None,
                          help='If you\'re using a pre-trained model and want the fine-tuned one to have a new name')
        self.add_argument("--seed", type=int, default=1,
                          help="random seed; if 0, a random random seed will be used  (default: 1)")
        self.add_argument("--task-id-seed", action='store_true',
                          help="use the task id within a Slurm job array as the seed")
        self.add_argument("--procs", type=int, default=64,
                          help="number of processes (default: 64)")
        self.add_argument("--tb", action="store_true", default=False,
                          help="log into Tensorboard")

        # Training arguments
        self.add_argument("--log-interval", type=int, default=10,
                          help="number of updates between two logs (default: 10)")
        self.add_argument("--frames", type=int, default=int(9e10),
                          help="number of frames of training (default: 9e10)")
        self.add_argument("--patience", type=int, default=100,
                          help="patience for early stopping (default: 100)")
        self.add_argument("--epochs", type=int, default=4)
        self.add_argument("--epoch-length", type=int, default=0,
                          help="number of examples per epoch; the whole dataset is used by if 0")
        self.add_argument("--frames-per-proc", type=int, default=40,
                          help="number of frames per process before update (default: 40)")
        self.add_argument("--beta1", type=float, default=0.9,
                          help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                          help="beta2 for Adam (default: 0.999)")
        self.add_argument("--optim-eps", type=float, default=1e-5,
                          help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
        self.add_argument("--optim-alpha", type=float, default=0.99,
                          help="RMSprop optimizer apha (default: 0.99)")
        self.add_argument("--batch-size", type=int, default=2048,
                          help="batch size for distillation")
        self.add_argument("--entropy-coef", type=float, default=0.1,
                          help="entropy term coefficient")

        # Model parameters
        self.add_argument("--image-dim", type=int, default=128,
                          help="dimensionality of the image embedding")
        self.add_argument("--memory-dim", type=int, default=128,
                          help="dimensionality of the memory LSTM")
        self.add_argument("--instr-dim", type=int, default=128,
                          help="dimensionality of the memory LSTM")
        self.add_argument("--no-instr", action="store_true", default=False,
                          help="don't use instructions in the model")
        self.add_argument("--instr-arch", default="gru",
                          help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
        self.add_argument("--arch", default='bow_endpool_res',
                          help="image embedding architecture")

        # Validation parameters
        self.add_argument("--val-seed", type=int, default=int(1e9),
                          help="seed for environment used for validation (default: 1e9)")
        self.add_argument("--val-interval", type=int, default=1,
                          help="number of epochs between two validation checks (default: 1)")
        self.add_argument("--val-episodes", type=int, default=500,
                          help="number of episodes used to evaluate the agent, and to evaluate validation accuracy")

        # NEW PARAMS!

        # Core params
        self.add_argument('--level', type=int, default=0)
        self.add_argument('--n_itr', type=int, default=100000)
        self.add_argument('--source', type=str, default='agent_probs', choices=['agent', 'teacher', 'agent_argmax',
                                                                                'agent_probs'])
        self.add_argument('--single_level', action='store_true')
        self.add_argument('--end_on_full_buffer', action='store_true')
        self.add_argument('--algo', type=str, default='ppo', choices=['sac', 'ppo', 'hppo'])

        # Saving/loading/finetuning
        self.add_argument('--prefix', type=str, default='DEBUG')
        self.add_argument('--description', type=str, default='yolo')
        self.add_argument('--save_option', type=str, default='level',
                          choices=['all', 'level', 'latest', 'none', 'gap'])

        # Meta
        self.add_argument('--reset_goal', action='store_true')
        self.add_argument('--reset_objs', action='store_true')
        self.add_argument('--reset_agent', action='store_true')
        self.add_argument('--rollouts_per_meta_task', type=int, default=1)

        # Teacher
        self.add_argument('--feedback_always', action='store_true')
        self.add_argument('--feedback_freq', nargs='+', type=int, default=[1])
        self.add_argument('--cartesian_steps', nargs='+', type=int, default=[1])
        self.add_argument('--teacher_schedule', type=str, default='all_teachers')
        self.add_argument('--use_dagger', action='store_true')
        self.add_argument('--collect_with_oracle', action='store_true')
        self.add_argument('--swap_factor', type=float, default=.5)
        self.add_argument('--include_zeros', action='store_true')
        self.add_argument('--success_intervention_cutoff', type=float, default=.95)
        self.add_argument('--accuracy_intervention_cutoff', type=float, default=.95)

        # Curriculum
        self.add_argument('--advance_curriculum_func', type=str, default='one_hot',
                          choices=["one_hot", "smooth", "uniform", 'four_levels', 'four_big_levels', 'five_levels',
                                   'goto_levels', 'easy_goto'])
        self.add_argument('--success_threshold_rl', type=float, default=1)
        self.add_argument('--success_threshold_rollout_teacher', type=float, default=1)
        self.add_argument('--success_threshold_rollout_no_teacher', type=float, default=1)
        self.add_argument('--accuracy_threshold_rl', type=float, default=.95)
        self.add_argument('--accuracy_threshold_distill_teacher', type=float, default=.9)
        self.add_argument('--accuracy_threshold_distill_no_teacher', type=float, default=.6)
        self.add_argument('--accuracy_threshold_rollout_teacher', type=float, default=.85)
        self.add_argument('--accuracy_threshold_rollout_no_teacher', type=float, default=.5)
        self.add_argument('--curriculum_type', type=int, default=1)
        self.add_argument('--augment', action='store_true')
        self.add_argument('--min_itr_steps', type=int, default=0)
        self.add_argument('--min_itr_steps_distill', type=int, default=0)
        self.add_argument('--advancement_count', type=int, default=1)

        # Model/Optimization
        self.add_argument('--lr', type=float, default=1e-4)
        self.add_argument('--discount', type=float, default=.25)
        self.add_argument('--num_modules', type=int, default=2)
        self.add_argument('--value_loss_coef', type=float, default=.05)  # .5 is default
        self.add_argument('--max_grad_norm', type=float, default=.5)
        self.add_argument('--clip_eps', type=float, default=.2)
        self.add_argument('--advice_dim', type=int, default=128)
        self.add_argument('--no_teacher', action='store_true')
        self.add_argument('--early_entropy_coef', type=float, default=None)
        self.add_argument('--fully_observed', action='store_true')

        # Reward
        self.add_argument('--reward_type', type=str, choices=['dense', 'sparse', 'oracle_action', 'oracle_dist',
                                                              'vector_dir', 'vector_dir2', 'vector_dir_final',
                                                              'vector_dir_waypoint', 'vector_dir_both',
                                                              'vector_dir_waypoint_negative', 'waypoint',
                                                              'vector_next_waypoint', 'wall_penalty', 'dense_pos_neg',
                                                              'dense_success'],
                          default='oracle_dist')
        self.add_argument('--ceil_reward', action='store_true')
        self.add_argument('--reward_when_necessary', action='store_true')

        # Policies
        self.add_argument('--collect_policy', default=None, help='path to collection policy')
        self.add_argument('--collect_teacher', default=None,
                          help="Teacher to for collection. If None, no collection happens. If no teacher, put 'None'")
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
        self.add_argument('--buffer_capacity', type=int, default=1000000)
        self.add_argument('--prob_current', type=float, default=.5)
        self.add_argument('--buffer_path', type=str, default=None)
        self.add_argument('--distill_dropout_prob', type=float, default=0.)
        self.add_argument('--collect_dropout_prob', type=float, default=0.)
        self.add_argument('--rollout_without_instrs', action='store_true')
        self.add_argument('--collect_before_threshold', action='store_true')
        self.add_argument('--distill_successful_only', action='store_true')
        self.add_argument('--kl_coef', type=float, default=0)
        self.add_argument('--control_penalty', type=float, default=0)
        self.add_argument('--mi_coef', type=float, default=0.01)
        self.add_argument('--z_dim', type=int, default=32)
        self.add_argument('--info_bot', action='store_true')

        # Arguments we rarely change
        self.add_argument('--meta_batch_size', type=int, default=200)
        self.add_argument('--sequential', action='store_true')
        self.add_argument('--max_path_length', type=float, default=float('inf'))
        self.add_argument('--gae_lambda', type=float, default=.99)
        self.add_argument('--num_envs', type=int, default=20)
        self.add_argument('--zero_all_thresholds', action='store_true')

        # Arguments mostly used with finetuning
        self.add_argument('--no_distill', action='store_true')
        self.add_argument('--yes_distill', action='store_true')
        self.add_argument('--leave_out_object', action='store_true')

        # Miscellaneous
        self.add_argument('--rollout_temperature', type=float, default=1)
        self.add_argument('--reconstruction', action='store_true')
        self.add_argument('--padding', action='store_true')
        self.add_argument('--feedback_from_buffer', action='store_true')
        self.add_argument('--same_model', action='store_true')
        self.add_argument('--rew_norm', action='store_true')
        self.add_argument('--act_norm', action='store_true')
        self.add_argument('--loss_type', type=str, default='log_prob')
        self.add_argument('--hidden_size', type=int, default=1024)
        self.add_argument('--early_stop', type=int, default=float('inf'))
        self.add_argument('--early_stop_metric', type=str, default=None)
        self.add_argument('--show_pos', type=str, choices=['ours', 'default', 'none'], default='none')
        self.add_argument('--show_goal', type=str, choices=['ours', 'offset', 'none'], default='none')
        self.add_argument('--show_agent_in_grid', action='store_true')
        self.add_argument('--reset_each_batch', action='store_true')
        self.add_argument('--no_buffer', action='store_true')
        self.add_argument('--static_env', action='store_true')
        self.add_argument('--save_untrained', action='store_true')
        self.add_argument('--reload_exp_path', type=str, default=None)
        self.add_argument('--continue_train', action='store_true')
        self.add_argument('--num_rollouts', type=int, default=10)
        self.add_argument('--eval_envs', nargs='+', type=int, default=None)
        self.add_argument('--hide_instrs', action='store_true')

    def parse_args(self, arg=None):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args(arg)

        # Set seed for all randomness sources
        if args.seed == 0:
            args.seed = np.random.randint(10000)
        if args.task_id_seed:
            args.seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
            print('set seed to {}'.format(args.seed))

        # TODO: more validation

        return args
