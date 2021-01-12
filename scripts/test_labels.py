import numpy as np
from babyai.utils.buffer import Buffer, trim_batch

buffer_file = "/home/olivia/Documents/Teachable/babyai/meta-mb-internal/data/T1250_CC3_DISILL_ORACLE_teacherPreActionAdviceCartesianCorrections_SD_threshS1_threshAR0.6_threshAD0.99_lr0.0001_ent0.0001_1"
buffer = Buffer(buffer_file, 1000, 1, val_prob=.1, augmenter=None)

correct = 0
total = 0
for level in range(25):
    print("starting level", level)
    for index in range(1000):
        try:
            traj = buffer.load_traj(level, index, 'train')
            for i in range(len(traj) - 3):
                timestep = traj[i]
                if timestep.obs['gave_CartesianCorrections']:
                    curr_cc = timestep.obs['CartesianCorrections']
                    pred_cc = traj[i + 3].obs['obs']
                    if not np.array_equal(curr_cc, pred_cc):
                        print("Uh oh!")
                    else:
                        correct += 1
                    total += 1
                    #     print(f"Okay on Level {level} index {index}")
        except Exception as e:
            break
print(correct/total)