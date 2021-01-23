import numpy as np
import cv2
import wandb
import os
import copy


def write_video(writer, frames, show_last=None):
    if show_last is not None:
        frames = frames[-show_last:]
    for frame in frames:
        writer.write(frame)


def write_traj_local(video_filename, curr_images, success, success_writer, failure_writer, all_writer, fps, size):
    # Add a few blank frames at the end to indicate success (white) or failure (black)
    sample_img = np.zeros_like(curr_images[-1])
    if success:
        curr_images += [sample_img + 255] * 3
        if success_writer is None:
            success_writer = cv2.VideoWriter(video_filename[:-4] + "success" + video_filename[-4:],
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        write_video(success_writer, curr_images)
    else:
        curr_images += [sample_img] * 3
        if failure_writer is None:
            failure_writer = cv2.VideoWriter(video_filename[:-4] + "failures" + video_filename[-4:],
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        write_video(failure_writer, curr_images)
    write_video(all_writer, curr_images)


def finalize_videos_local(video_filename, all_writer, success_writer, failure_writer):
    all_writer.release()
    if success_writer is not None:
        success_writer.release()
    if failure_writer is not None:
        failure_writer.release()
    print("Video saved at %s" % video_filename)


def finalize_videos_wandb(video_name, all_videos, success_videos, failure_videos, fps):
    video = np.transpose(np.stack(all_videos), (0, 3, 1, 2))[:, ::-1]
    wandb.log({video_name + '_all': wandb.Video(video, fps=fps, format="mp4")}, commit=False)
    if len(success_videos) > 0:
        video = np.transpose(np.stack(success_videos), (0, 3, 1, 2))[:, ::-1]
        wandb.log({video_name + '_success': wandb.Video(video, fps=fps, format="mp4")}, commit=False)
    if len(failure_videos) > 0:
        video = np.transpose(np.stack(failure_videos), (0, 3, 1, 2))[:, ::-1]
        wandb.log({video_name + '_failure': wandb.Video(video, fps=fps, format="mp4")}, commit=False)


def plot_img(env, agent_action, teacher_action, record_teacher, run_index, teacher_name):
    # TODO: if we reintroduce the reward predictor, plot it here too
    image = env.render(mode='rgb_array')[:, :, ::-1]  # RGB --> BGR
    h, w, c = image.shape
    background = np.zeros((h * 2, w * 2, c), dtype=np.uint8) + 255
    if not agent_action == teacher_action:
        background[:, :, 0] = 0
    background[h:, w:] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_str = ""
    if hasattr(env, "teacher") and env.teacher is not None:
        if record_teacher:
            label_str += f"Teacher advice: {teacher_action}"
        else:
            label_str += f"Teacher on: {env.teacher.feedback_type}   "
    label_str += "Run: " + str(run_index)
    cv2.putText(background, env.mission, (30, 30), font, 0.5, (0, 0, 0), 1, 0)
    cv2.putText(background, label_str, (30, 90), font, 0.5, (0, 0, 0), 1, 0)
    cv2.putText(background, "Action " + str(agent_action), (30, 60), font, 0.5, (0, 0, 0), 1, 0)
    cv2.putText(background, "Receiving Teacher " + teacher_name, (30, 120), font, 0.5, (0, 0, 0), 1, 0)
    return background


def check_followed_cc3(obs_list):
    if not 'CartesianCorrections' in obs_list[0]:
        return 0
    gave_cc3 = 0
    followed_cc3 = 0
    for i in range(len(obs_list) - 3):
        obs = obs_list[i]
        if obs['gave_CartesianCorrections']:
            gave_cc3 += 1
            if np.array_equal(obs_list[i + 3]['obs'], obs_list[i]['CartesianCorrections']):
                followed_cc3 += 1
    if gave_cc3 == 0:
        return 0
    return followed_cc3 / gave_cc3


def rollout(env, agent, instrs=True, max_path_length=np.inf, speedup=1, reset_every=1,
            video_directory="", video_name='sim_out', stochastic=False, num_rollouts=1,
            num_save=None, record_teacher=False, reward_predictor=None, save_locally=True,
            save_wandb=False, obs_preprocessor=None, teacher_dict={}, teacher_name="", rollout_oracle=False,
            temperature=1):
    video_filename = os.path.join(video_directory, video_name + ".mp4")
    if num_save is None:
        num_save = num_rollouts

    # Get setup to log
    timestep = getattr(env, "dt", 0.5)
    fps = int(speedup / timestep)
    img = env.render(mode='rgb_array')
    height, width, channels = img.shape
    size = (width * 2, height * 2)
    if save_locally:
        all_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        success_writer = None
        failure_writer = None
    if save_wandb:
        all_videos, success_videos, failure_videos = [], [], []

    # Collect a few trajectories
    paths, agent_actions, teacher_actions = [], [], []
    correct, stoch_correct, det_correct, count = 0, 0, 0, 0
    full_obs_list = []
    for i in range(num_rollouts):
        observations, actions, rewards, agent_infos, env_infos, curr_images = [], [], [], [], [], []
        path_length = 0

        # Possibly reset the env. If we're doing RL2, we may not reset every time.
        if i % reset_every == 0:
            agent.reset(dones=[True])
            if reward_predictor is not None:
                reward_predictor.reset(dones=[True])
            env.set_task()
        o = env.reset()

        # Loop until the max_path_length or we hit done
        while path_length < max_path_length:
            full_obs_list.append(copy.deepcopy(o))
            # Choose action
            o = obs_preprocessor([o], teacher_dict, show_instrs=instrs)
            a, agent_info = agent.get_actions_t(o, temp=temperature)

            a = a.item()
            stoch_a = a
            det_a = np.argmax(agent_info[0]['probs'])
            if not stochastic:
                a = np.argmax(agent_info[0]['probs'])

            # Step env
            if rollout_oracle:
                next_o, r, d, env_info = env.step(env.teacher_action)
            else:
                next_o, r, d, env_info = env.step(a)

            # use reward predictor
            if reward_predictor is not None:  # TODO: we currently don't do anything with this!
                reward_obs = np.stack([env_info['next_obs_rewardfree']])
                pred_reward = reward_predictor.get_actions_t(reward_obs)

            # Store data for logging
            success = env_info['success']
            teacher_actions.append(env_info['teacher_action'])
            if env_info['teacher_action'] == a:
                correct += 1
            if env_info['teacher_action'] == stoch_a:
                stoch_correct += 1
            if env_info['teacher_action'] == det_a:
                det_correct += 1
            count += 1
            agent_actions.append(a)
            observations.append(o)
            rewards.append(r)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1

            o = next_o

            # Render image, if necessary
            if (save_locally or save_wandb) and i < num_save:
                img = plot_img(env, agent_action=a, teacher_action=env_info['teacher_action'],
                               record_teacher=record_teacher, run_index=i % reset_every, teacher_name=teacher_name)
                curr_images.append(img)

            # End trajectory on 'done'
            if d:
                break

        # At the end of a trajectory, save it
        if save_locally and i < num_save:
            write_traj_local(video_filename, curr_images, success, success_writer, failure_writer,
                             all_writer, fps, size)
        if save_wandb and i < num_save:
            sample_img = np.zeros_like(curr_images[-1])
            all_videos += curr_images
            if success:
                success_videos += curr_images + [sample_img + 255] * 3
            else:
                failure_videos += curr_images + [sample_img + 255] * 3

        paths.append(dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            agent_infos=agent_infos,
            env_infos=env_infos
        ))

    # Finish saving videos
    if save_locally:
        finalize_videos_local(video_filename, all_writer, success_writer, failure_writer)
    if save_wandb:
        finalize_videos_wandb(video_name, all_videos, success_videos, failure_videos, fps)

    followed_cc3_proportion = check_followed_cc3(full_obs_list)
    return paths, correct / count, stoch_correct / count, det_correct / count, followed_cc3_proportion
