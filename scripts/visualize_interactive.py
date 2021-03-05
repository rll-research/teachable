import joblib
import tensorflow as tf
import numpy as np
import argparse
import copy
from matplotlib import pyplot as plt
import torch
from babyai.utils.obs_preprocessor import make_obs_preprocessor
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--level", type=int, default=None)
parser.add_argument("--supervised", type=bool, default=False)
args = parser.parse_args()


def pa_modify(advice, obs):
    num = int(advice)
    obs['PreActionAdvice'] = np.zeros_like(obs['PreActionAdvice'])
    obs['PreActionAdvice'][num] = 1
    return obs


def cc3_modify(advice, obs, env):
    env_copy = pkl.loads(pkl.dumps(env))
    for action in advice:
        action = int(action)
        o, r, d, i = env_copy.step(action)
    advice = o['obs'].flatten()
    obs['CartesianCorrections'] = advice
    return obs


# with tf.Session() as sess:
if True:
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

    teacher_dict = {'CartesianCorrections': True}
    teacher_dict = {'PreActionAdvice': False, 'CartesianCorrections': True}
    # teacher_dict = {'PreActionAdvice': True, 'CartesianCorrections': False}
    teacher_name = 'cc3'
    # teacher_name = 'pa'

    # Turn left policy
    base_path = "/home/olivia/Documents/Teachable/babyai/meta-mb-internal/data/"
    pkl_path = args.path

    pkl_path = base_path + pkl_path
    data = joblib.load(pkl_path)
    if args.supervised:
        agent = data['supervised_model']
    else:
        agent = data['policy']
    # agent.eval()
    env = data['env']
    if args.level is not None:
        env.set_level_distribution(args.level)

    # env = env._wrapped_env._wrapped_env._wrapped_env
    env = env.held_out_levels[1]
    print(type(env))

    env.set_task(None)

    obs_orig = env.reset()
    count = 0
    skip_to_done = False
    skip_to_error = False
    teacher_recommendations = []
    agent_actions = []
    # hidden_state = torch.zeros([1, agent.memory_size], device=agent.device)
    teacher_null_dict = env.teacher.null_feedback()
    obs_preprocessor = make_obs_preprocessor(teacher_null_dict)
    skip_time = False
    while True:
        # Interaction
        ready = ""
        teacher_action = env.teacher_action.item()
        try:
            obs_processed = obs_preprocessor([obs_orig], teacher_dict)
        except:
            obs_processed = obs
            x = 3
        # dist, agent_info = agent(obs_processed)
        # agent_action = torch.argmax(dist.probs, dim=1).item()
        # print("The Teacher is saying to take", teacher_action)
        # print("The Agent is most likely to take", agent_action)
        # error = not teacher_action == agent_action
        while (not skip_to_done):
            print("please type a command below")
            ready = input()
            if ready in ['h', 'help']:
                print("Type 'h' for help, 'c' to continue to the next timestep, 'm' to continue with the modified "
                      "version of the observation with the alt teacher action (note: if none has been specified, it"
                      "functions the same as 'c'.  Type a number to make the teacher suggest the action at that index.")
            elif ready in ['r', 'render']:
                # img = env.render(mode='rgb_array')
                # plt.imshow(img)
                # plt.title(env.mission)
                # plt.show()
                env.render(mode='human')
            elif ready in ['c', 'continue']:
                obs = copy.deepcopy(obs_orig)
                obs = obs_preprocessor([obs], teacher_dict)
                break
            elif ready in ['sd', 'skip_done']:
                skip_to_done = True
                break
            elif ready in ['e', 'a', 'error']:
                error = not teacher_action == agent_action
                i = 0
                while not error:
                    print("step", i)
                    i += 1
                    obs_orig, r, done, env_info = env.step(agent_action)
                    if done:
                        obs_orig = env.reset()
                    obs = obs_preprocessor([copy.deepcopy(obs_orig)], teacher_dict)
                    dist, agent_info = agent(obs, hidden_state)
                    teacher_action = env.teacher_action.item()
                    if ready == 'e':
                        agent_action = dist.sample()
                    else:
                        agent_action = torch.argmax(dist.probs, dim=1).item()
                    error = not teacher_action == agent_action
                print("ERROR!", "teacher", teacher_action, "agent", agent_action)
                img = env.render(mode='rgb_array')
                plt.imshow(img)
                plt.title(env.mission)
                plt.show()
                skip_to_error = False
                skip_time = True
                break
            elif ready in ['m', 'modified']:
                obs = modified_obs
                break
            elif ready in ['n', 'normal']:
                obs_processed = obs_preprocessor([obs_orig], teacher_dict)
                dist, agent_info = agent(obs_processed, hidden_state)
                a = dist.sample().item()
                probs = np.round(dist.probs.detach().cpu().numpy(), 2)[0]
                print("Most likely action is", a, 'distribution is', probs)
            else:
                try:
                    if teacher_name == 'pa':
                        modified_obs = pa_modify(ready, obs_orig)
                    elif teacher_name == 'cc3':
                        modified_obs = cc3_modify(ready, obs_orig, env)
                    print("MODIFIED Teacher suggested:", ready)
                    modified_obs = obs_preprocessor([modified_obs], teacher_dict)
                    dist, agent_info = agent(modified_obs, hidden_state)
                    a = dist.sample().item()
                    probs = np.round(dist.probs.detach().cpu().numpy(), 2)[0]
                    print("MODIFIED most likely action is", a, 'distribution is', probs)
                except Exception as e:
                    print(e)
                    print("Invalid index", ready)

        # Advance env
        if not skip_time:
            try:
                dist, agent_info = agent(obs, hidden_state)
            except:
                obs = obs_preprocessor([obs], teacher_dict)
                dist, agent_info = agent(obs, hidden_state)

            a = dist.sample()
            hidden_state = agent_info['memory']
            obs_orig, r, d, env_info = env.step(a)
            # print("Teacher suggested:", env_info['teacher_action'])
            # teacher_recommendations.append(np.argmax(obs[0, 160:168]))
            print("Agent took", a.item(), dist.probs)
            # agent_actions.append(a.item())
            print("Done?", d)
            ready = ''

            print("Success?", r)
            if d:
                skip_to_done = False
                env.set_task(None)
                obs = env.reset()
                count += 1

                # plt.figure(1)
                # plt.hist([teacher_recommendations, agent_actions], bins=list(range(len(action_meanings))),
                #          label=action_meanings, color=["blue", "orange"])
                # plt.legend(["Teacher Suggested", "Agent took"])
                # plt.title("Teacher's suggestions vs agent's actions")
                # plt.show()

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
        skip_time = False
