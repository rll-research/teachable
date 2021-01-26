import argparse
import os
import pathlib

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--envs', nargs='+', type=str)
parser.add_argument('--teachers', nargs='+', type=str, choices=['first', 'last', 'none', 'all'], default=['all'])
parser.add_argument('--policies', nargs='+', required=True, type=str)
parser.add_argument("--num_rollouts", type=int, default=10)
parser.add_argument("--starting_index", type=int, default=0)
parser.add_argument("--finetune_itrs", type=int, default=10)
parser.add_argument("--log_every", type=int, default=1)
parser.add_argument("--first_teacher", type=str, default='PreActionAdvice')
parser.add_argument("--last_teacher", type=str, default='PreActionAdviceMultiple')
parser.add_argument("--finetune_envs_together", action='store_true')
parser.add_argument("--generate_eval", action='store_true')
parser.add_argument("--generate_finetune", action='store_true')
parser.add_argument("--abstract_teacher_multiplier", type=int, default=3)
args = parser.parse_args()
if 'all' in args.teachers:
    args.teachers = ['first', 'last', 'none']

def get_policy_path(s):
    base_path = pathlib.Path(os.getcwd()).joinpath('data')
    runs = []
    for run in base_path.iterdir():
        if s in run.name:
            runs.append(run)
    assert len(runs) == 1, ("Must have single run", s, runs)
    return runs[0]

def get_teacher_name(teacher):
    if teacher == 'first':
        teacher_name = args.first_teacher
    elif teacher == 'last':
        teacher_name = args.last_teacher
    else:
        assert teacher == 'none', f"unrecognized teacher {teacher}"
        teacher_name = teacher
    return teacher_name

index = args.starting_index

def get_command(policy, envs, itrs, teacher, rollouts, log_every, no_train_rl=False, teacher_schedule=None,
                distillation_strategy=None, yes_distill=None, no_distill=None, finetune_il=None):
    policy_path = get_policy_path(policy)
    policy_name = policy[1:5] if policy[0] == 'T' else policy[:4] # Assumes the policy name starts with Txxx or xxxx.
    env_str = ' '.join(envs)
    global index
    s = f'CUDA_VISIBLE_DEVICES=X python ../scripts/test_generalization.py ' \
        f'--policy "{policy_path}" ' \
        f'--envs {env_str} ' \
        f'--save_dir eval_runs/T{index:04d}_EVAL_{policy_name}_FINETUNE{itrs}_INSTR_{teacher} ' \
        f'--finetune_itrs {itrs} ' \
        f'--teachers {teacher} ' \
        f'--num_rollouts {rollouts} ' \
        f'--log_every {log_every} '
    if no_train_rl:
        s += f'--no_train_rl '
    if teacher_schedule is not None:
        s += f'--teacher_schedule {teacher_schedule} '
    if distillation_strategy is not None:
        s += f'--distillation_strategy {distillation_strategy} '
    if yes_distill:
        s += f'--yes_distill '
    if no_distill:
        s += f'--no_distill '
    if finetune_il:
        s += f'--finetune_il '
    index += 1
    return s

if args.generate_eval:
    for policy in args.policies:
        for teacher in args.teachers:
            teacher_name = get_teacher_name(teacher)
            s = get_command(policy, args.envs, 0, teacher_name, args.num_rollouts, 0)
            print(s)
if args.generate_finetune:
    envs = args.envs
    if args.finetune_envs_together:
        envs = [envs]  # Wrap in a list so they're put together into the same command
    for env in envs:
        for policy in args.policies:
            for teacher in args.teachers:
                if teacher == 'first':
                    no_train_rl = True
                    teacher_schedule = 'first_teacher'
                    distillation_strategy = 'no_teachers'
                    yes_distill = True
                    no_distill = False
                elif teacher == 'last':
                    no_train_rl = True
                    teacher_schedule = 'last_teacher'
                    distillation_strategy = 'no_teachers'
                    yes_distill = True
                    no_distill = False
                elif teacher == 'none':
                    no_train_rl = False
                    teacher_schedule = 'none'
                    distillation_strategy = None
                    yes_distill = False
                    no_distill = True
                else:
                    raise NotImplementedError(teacher)
                finetune_itrs = args.finetune_itrs
                if teacher == 'last':
                    finetune_itrs *= args.abstract_teacher_multiplier
                s = get_command(policy, args.envs, finetune_itrs, 'none', args.num_rollouts, args.log_every,
                                no_train_rl=no_train_rl, teacher_schedule=teacher_schedule,
                                distillation_strategy=distillation_strategy,
                                yes_distill=yes_distill, no_distill=no_distill)
                print(s)