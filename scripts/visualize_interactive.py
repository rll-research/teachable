import joblib
import tensorflow as tf
import numpy as np
import argparse
import copy
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--level", type=int, default=None)
parser.add_argument("--supervised", type=bool, default=False)
args = parser.parse_args()

# EXAMPLES

# Policy which generally follows the teacher, except that it turns right when it says turn left.
# If the teacher said something different in that situation, the agent would follow if it's a common command (straight
# or right), but if it was an uncommon command like toggling, it would ignore the teacher.
# SUPERVISED_teacherPreActionAdvice_persistgoa_droptypestep_dropgoal0_ent0.001_lr0.01corr0_currfnsmooth_4/latest.pkl

# (seems similar to the policy above)
# THRESHOLD++_teacherPreActionAdvice_persistgoa_droptypestep_dropincNone_dropgoal0_disc0.9_thresh0.99_ent0.001_lr0.01corr0_currfnsmooth_4

# Kinda sorta follows the teacher, but gets stuck on toggling. Consider more entropy???
# "THRESHOLD++_teacherPreActionAdvice_persistgoa_droptypemeta_rollout_start_dropinc(0.8, 0.2)_dropgoal0_disc0.9_thresh0.95_ent0.001_lr0.01corr0_currfnsmooth_4/latest.pkl"




with tf.Session() as sess:

    # ACTION SPACE
    # left = 0
    # right = 1
    # forward = 2
    # pickup = 3
    # drop = 4
    # toggle = 5
    # done = 6
    # no feedback = 7 // -1
    action_meanings = ["left", "right", "forward", "pickup", "drop", "toggle", "done", "no feedback"]

    # Turn left policy
    base_path = "/home/olivia/Documents/Teachable/babyai/meta-mb-internal/data/"
    pkl_path = args.path

    pkl_path = base_path + pkl_path
    data = joblib.load(pkl_path)
    if args.supervised:
        agent = data['supervised_model']
    else:
        agent = data['policy']
    env = data['env']
    if args.level is not None:
        env.set_level_distribution(args.level)

    agent.reset(dones=[True])
    env.set_task(None)

    env.set_dropout_proportion(1)
    obs = env.reset()
    obs = np.expand_dims(obs, 0)

    use_one_hot = True

    count = 0
    skip_to_done = False
    skip_to_meta_rollout = False
    teacher_recommendations = []
    agent_actions = []
    while True:
        # Interaction
        ready = ""
        hidden_state = copy.deepcopy(agent._hidden_state)
        obs_orig = copy.deepcopy(obs)
        while (not skip_to_done) and (not skip_to_meta_rollout):
            print("please type a command below")
            ready = input()
            if ready in ['h', 'help']:
                print("Type 'h' for help, 'c' to continue to the next timestep, 'm' to continue with the modified "
                      "version of the observation with the alt teacher action (note: if none has been specified, it"
                      "functions the same as 'c'.  Type a number to make the teacher suggest the action at that index.")
            elif ready in ['r', 'render']:
                env.render(mode='human')
            elif ready in ['c', 'continue']:
                break
            elif ready in ['sd', 'skip_done']:
                skip_to_done = True
                break
            elif ready in ['sm', 'skip_meta']:
                skip_to_meta_rollout = True
                break
            elif ready in ['m', 'modified']:
                obs_orig = obs
                break
            else:
                try:
                    agent.set_hidden_state(hidden_state)
                    num = int(ready)
                    assert num >= 0
                    assert num <= 6
                    agent.reset(dones=[True])
                    if use_one_hot:
                        obs[0, 160:168] = 0
                        obs[0, 160 + num] = 1
                        print("MODIFIED Teacher suggested:", np.argmax(obs[0, 160:168]))
                    else:
                        if num == 7:
                            num = -1
                        obs[0, 160] = num
                        print("MODIFIED Teacher suggested:", obs[0, 160])
                    a, agent_info = agent.get_actions(obs)
                    a = a[0]
                    print("MODIFIED most likely action is", np.argmax(agent_info[0][0]['probs']), agent_info[0][0]['probs'])
                except:
                    print("Invalid index", ready)

        # Advance env
        obs = obs_orig
        agent.set_hidden_state(hidden_state)
        a, agent_info = agent.get_actions(obs)
        a = a[0]
        if use_one_hot:
            print("Teacher suggested:", np.argmax(obs[0, 160:168]))
            teacher_recommendations.append(np.argmax(obs[0, 160:168]))
        else:
            print("Teacher suggested:", obs[0, 160])
            teacher_recommendations.append(obs[0, 160])
        if obs[0, 160] == 6:
            print("what????")
        print("Agent took", a[0][0], agent_info[0][0]['probs'])
        agent_actions.append(a[0][0])
        obs, r, d, env_info = env.step(a)
        obs = np.expand_dims(obs, 0)
        print("Done?", d)
        ready = ''

        print("Success?", r)
        if d:
            skip_to_done = False
            if count % 2 == 1:
                agent.reset(dones=[True])
                env.set_task(None)
                skip_to_meta_rollout = False
            obs = env.reset()
            obs = np.expand_dims(obs, 0)
            count += 1

            plt.figure(1)
            plt.hist([teacher_recommendations, agent_actions], bins=list(range(len(action_meanings))), label=action_meanings, color=["blue", "orange"])
            plt.legend(["Teacher Suggested", "Agent took"])
            plt.title("Teacher's suggestions vs agent's actions")
            plt.show()

            agent_actions = np.array(agent_actions)
            teacher_recommendations = np.array(teacher_recommendations)
            percent_listened = []
            for i in range(len(action_meanings)):
                indices = np.where(teacher_recommendations == i)
                actions = agent_actions[indices]
                proportion_correct = np.mean(actions == i)
                percent_listened.append(proportion_correct)

            plt.figure(3)
            percent_listened = [0 if np.isnan(p) else p for p in percent_listened]
            plt.bar(action_meanings, percent_listened)
            plt.title("Proportion of the time each action was followed")
            plt.show()

            teacher_recommendations = []
            agent_actions = []
