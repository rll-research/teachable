import numpy as np
from babyai.utils.buffer import Buffer, trim_batch
import joblib
import matplotlib.pyplot as plt
from babyai.rl.utils.dictlist import DictList
import pickle as pkl
from babyai.levels.curriculum import Curriculum
from meta_mb.meta_envs.rl2_env import rl2env
from meta_mb.envs.normalized_env import normalize

use_buffer = False

buffer_file = "/home/olivia/Documents/Teachable/babyai/meta-mb-internal/data/T1250_CC3_DISILL_ORACLE_teacherPreActionAdviceCartesianCorrections_SD_threshS1_threshAR0.6_threshAD0.99_lr0.0001_ent0.0001_1"
# buffer_file = "/home/olivia/Teachable/babyai/meta-mb-internal/data/T1369_COLLECT_teacherPreActionAdviceCartesianCorrections_threshS1_threshAR0.95_threshAD0.95_lr0.0001_ent0.0001_1"
if use_buffer:
    buffer = Buffer(buffer_file, 1000, 1, val_prob=.1, augmenter=None)#, buffer_name='dataset'
else:
    # env2 = joblib.load(buffer_file + '/latest.pkl')['env']
    arguments = {
        "start_loc": 'all',
        "include_holdout_obj": True,
        "persist_goal": True,
        "persist_objs": True,
        "persist_agent": True,
        "feedback_type": ['PreActionAdviceMultiple', 'CartesianCorrections', 'XYCorrections', 'OffsetCorrections',
                          'PreActionAdvice', 'SubgoalCorrections'],
        "feedback_freq": [1],
        "cartesian_steps": 2,
        "num_meta_tasks": 1,
        "intermediate_reward": True,
    }
    env = rl2env(normalize(Curriculum('one_hot', start_index=0, curriculum_type=1, **arguments)), ceil_reward=False)
    env.reset()

succeeded = 0
trajs = 0
correct = 0
total = 0
level_total = 0
level_rare_tokens = 0
level_trajs = 0
for level in range(5, 25):
    if level > 0:
        if level_total > 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>level correct", level_correct / level_total)
            if level_correct < level_total:
                print("not again :(")
        if level_trajs > 0:
            print(">>>>>>>>>>>>>>>>level succeeded", level_succeeded / level_trajs)
            if level_succeeded < level_trajs:
                print("not again :(")
        if level_rare_tokens > 0:
            print("=================Rare messed up", level_messed_up_rare / level_rare_tokens)
    print("starting level", level)
    level_correct = 0
    level_total = 0
    level_messed_up_rare = 0
    level_rare_tokens = 0
    level_succeeded = 0
    level_trajs = 0
    for index in range(200):
        if index % 10 == 0:
            print(index)
        if not use_buffer:
            env.set_task(None)
            level_ = env.levels_list[level]
            level_env = env.levels_list[level]
            obs = level_env.reset()
            env_copy = pkl.loads(pkl.dumps(level_env))
            obs_list = []
            teacher_action_list = []
            teacher_action_list2 = []
            agent_action_list = []
            img_list = []
            done = False
            env_list = []
            instruction = level_env.mission
            while not done:
                env_list.append(pkl.loads(pkl.dumps(level_env)))
                obs_list.append(obs)
                action = level_env.teacher_action.item()
                teacher_action_list.append(int(list(level_env.teacher.teachers.values())[0].next_action))
                action = level_env.teacher_action.item()
                # action = np.random.choice(7, p=[.2, .2, .3, .1, .1, .1, 0])
                agent_action_list.append(action)
                img_list.append(level_env.render('rgb_array'))
                obs, r, done, info = level_env.step(action)
            level_trajs += 1
            level_succeeded += info['success']
            traj = DictList({'obs': obs_list, 'teacher_action': np.expand_dims(np.array(teacher_action_list), 1)})
        else:
            try:
                traj = buffer.load_traj(level, index, 'train')
            except Exception as e:
                break
        level_rare_tokens += np.sum(traj.teacher_action == 3)
        level_rare_tokens += np.sum(traj.teacher_action == 4)
        for i in range(len(traj) - 3):
            timestep = traj[i]
            steps = arguments['cartesian_steps']
            bad = False
            if timestep.obs['gave_PreActionAdviceMultiple']:
                curr_cc = timestep.obs['PreActionAdviceMultiple']
                curr_cc = np.array([np.argmax(curr_cc[k * 8 :(k + 1) * 8]) for k in range(steps)])
                pred_cc = traj[i:i + steps].teacher_action[:, 0]
                pa_bad = not np.array_equal(curr_cc, pred_cc)
                bad = bad or not np.array_equal(curr_cc, pred_cc)
            if timestep.obs['gave_CartesianCorrections']:
                curr_cc = timestep.obs['CartesianCorrections']
                pred_cc = traj[i + steps].obs['obs']
                bad = bad or not np.array_equal(curr_cc, pred_cc)
                cc3_bad = not np.array_equal(curr_cc, pred_cc)
            if bad:
                level_messed_up_rare += np.sum(traj.teacher_action[i:i+steps, 0] == 3)
                level_messed_up_rare += np.sum(traj.teacher_action[i:i+steps, 0] == 4)
                hints = [(np.argmax(t['PreActionAdviceMultiple'][:8]), np.argmax(t['PreActionAdviceMultiple'][8:])) for
                         t in traj.obs]
                print("Uh oh!", traj.teacher_action[i:i+steps, 0], f"{i}/ {len(traj)}")
                temp = 3
            else:
                level_correct += 1
                correct += 1
            level_total += 1
            total += 1

if level_total > 0:
    print(">>>>>>>>>>>>>>>>level correct", level_correct / level_total)
if level_trajs > 0:
    print(">>>>>>>>>>>>>>>>level succeeded", level_succeeded / level_trajs)
if level_rare_tokens > 0:
    print("==============rare messed up", level_messed_up_rare / level_rare_tokens)
print("overall correct", correct/total)
