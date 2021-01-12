import numpy as np
from babyai.utils.buffer import Buffer, trim_batch

buffer_file = "/home/olivia/Documents/Teachable/babyai/meta-mb-internal/data/T1250_CC3_DISILL_ORACLE_teacherPreActionAdviceCartesianCorrections_SD_threshS1_threshAR0.6_threshAD0.99_lr0.0001_ent0.0001_1"
buffer_file = "/home/olivia/Teachable/babyai/meta-mb-internal/data/T1369_COLLECT_teacherPreActionAdviceCartesianCorrections_threshS1_threshAR0.95_threshAD0.95_lr0.0001_ent0.0001_1"
buffer = Buffer(buffer_file, 1000, 1, val_prob=.1, augmenter=None, buffer_name='dataset')

correct = 0
total = 0
for level in range(25):
    print("starting level", level)
    if level > 0:
        print("level correct", level_correct / level_total)
        if level_rare_tokens > 0:
            print("Rare messed up", level_messed_up_rare / level_rare_tokens)
    level_correct = 0
    level_total = 0
    level_messed_up_rare = 0
    level_rare_tokens = 0
    for index in range(1000):
        try:
            traj = buffer.load_traj(level, index, 'train')
            level_rare_tokens += np.sum(traj.teacher_action == 3)
            level_rare_tokens += np.sum(traj.teacher_action == 4)
            for i in range(len(traj) - 3):
                timestep = traj[i]
                if timestep.obs['gave_CartesianCorrections']:
                    curr_cc = timestep.obs['CartesianCorrections']
                    pred_cc = traj[i + 3].obs['obs']
                    if not np.array_equal(curr_cc, pred_cc):
                        level_messed_up_rare += np.sum(traj.teacher_action[i:i+3, 0] == 3)
                        level_messed_up_rare += np.sum(traj.teacher_action[i:i+3, 0] == 4)
                        #print("Uh oh!", traj.teacher_action[i:i+3, 0])
                    else:
                        level_correct += 1
                        correct += 1
                    level_total += 1
                    total += 1
                    #     print(f"Okay on Level {level} index {index}")
        except Exception as e:
            print(e)
            break
print("level correct", level_correct / level_total)
if level_rare_tokens > 0:
    print("rare messed up", level_messed_up_rare / level_rare_tokens)
print("overall correct", correct/total)
