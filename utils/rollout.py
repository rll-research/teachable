import numpy as np
import cv2
# import wandb
import os
import copy
from gym_minigrid.minigrid import COLOR_NAMES
from gym.spaces import Discrete
OBJ_TYPES = ['box', 'ball', 'key', 'door']

def write_video(writer, frames, show_last=None):
    if show_last is not None:
        frames = frames[-show_last:]
    for frame in frames:
        writer.write(frame)


def write_traj_local(video_filename, curr_images, success, success_writer, failure_writer, all_writer, fps, size, codec):
    # Add a few blank frames at the end to indicate success (white) or failure (black)
    sample_img = np.zeros_like(curr_images[-1])
    if success:
        curr_images += [sample_img + 255] * 3
        if success_writer is None:
            success_writer = cv2.VideoWriter(video_filename[:-4] + "success" + video_filename[-4:],
                                             cv2.VideoWriter_fourcc(*codec), fps, size)
        write_video(success_writer, curr_images)
    else:
        curr_images += [sample_img] * 3
        if failure_writer is None:
            failure_writer = cv2.VideoWriter(video_filename[:-4] + "failures" + video_filename[-4:],
                                             cv2.VideoWriter_fourcc(*codec), fps, size)
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
    wandb.log_rollouts({video_name + '_all': wandb.Video(video, fps=fps, format="mp4")}, commit=False)
    if len(success_videos) > 0:
        video = np.transpose(np.stack(success_videos), (0, 3, 1, 2))[:, ::-1]
        wandb.log_rollouts({video_name + '_success': wandb.Video(video, fps=fps, format="mp4")}, commit=False)
    if len(failure_videos) > 0:
        video = np.transpose(np.stack(failure_videos), (0, 3, 1, 2))[:, ::-1]
        wandb.log_rollouts({video_name + '_failure': wandb.Video(video, fps=fps, format="mp4")}, commit=False)


def get_readable_feedback(env_info, obs, teacher_name):
    if teacher_name == 'Direction':
        return obs['Direction']
    if teacher_name == 'Waypoint':
        return obs['Waypoint']
    if teacher_name == 'Cardinal':
        feedback = np.argmax(obs['Cardinal'])
        return ['Left', 'Up', 'Right', 'Down'][feedback]
    if teacher_name == 'PreActionAdvice':
        return str(env_info['teacher_action'].item())
    if teacher_name in ['OFFSparseRandom', 'OSREasy', 'OSRPeriodicImplicit']:
        offset = obs[teacher_name]
        first = offset[0]
        coords_offset = offset[1:3]
        start_str = "Using an obj at " if first else "Going to"
        agent_pos = offset[3: 5] * 12 + 12
        agent_dir = offset[5] * 3
        if agent_dir < 0:
            agent_dir = offset[5]
            agent_pos = offset[3: 5]
        timesteps_ago = np.argmax(offset[6:])
        return f"{start_str} {coords_offset}, {timesteps_ago + 1}, pos {agent_pos}, dir {agent_dir}"
    elif teacher_name in ['SubgoalCorrections', 'SubgoalSimple']:
        subgoal_names = ['OpenSubgoal',
                         'DropSubgoal',
                         'PickupSubgoal',
                         'GoNextToSubgoal']
        subgoal = obs[teacher_name]
        # Subgoal Name
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
        coordinate = (coordinate * 10) + agent_pos
        return f"Name: {subgoal_name}, Coord: {coordinate}, " \
               f"obj {obj_color} {obj_type}, pos {agent_pos}, dir {agent_dir}"
    return f'no feedback string available for {teacher_name}'


def plot_img(env, obs, image, agent_action, env_info, record_teacher, teacher_name, reward):
    image = image[:, :, ::-1]  # RGB --> BGR
    return image


def plot_img_intermediate(env, obs, image, agent_action, env_info, record_teacher, teacher_name, reward):
    h, w, c = image.shape
    image = image[:, :, ::-1]  # RGB --> BGR
    background = np.zeros((h + 100, w, c), dtype=np.uint8) + 255
    background[:h] = image
    if hasattr(env, 'mission'):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(background, env.mission, (30, h + 30), font, 0.5, (0, 0, 0), 1, 0)
    return image


def plot_img_complex(env, obs, image, agent_action, env_info, record_teacher, teacher_name, reward):
    feedback = get_readable_feedback(env_info, obs, teacher_name)
    teacher_action = env_info['teacher_action']
    # TODO: if we reintroduce the reward predictor, plot it here too
    image = image[:, :, ::-1]  # RGB --> BGR
    h, w, c = image.shape
    background = np.zeros((h * 2, w * 2, c), dtype=np.uint8) + 255
    if type(agent_action) is int and not agent_action == teacher_action.item():  # TODO: handle continuous case better
        background[:, :, 0] = 0
    background[h:, w:] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_str = ""
    if hasattr(env, "teacher") and env.teacher is not None:
        if record_teacher:
            label_str += f"Teacher advice: {teacher_action}"
        else:
            label_str += f"Teacher on: {env.teacher.feedback_type}   "
    if hasattr(env, 'mission'):
        cv2.putText(background, env.mission, (30, 30), font, 0.5, (0, 0, 0), 1, 0)
    cv2.putText(background, label_str, (30, 90), font, 0.5, (0, 0, 0), 1, 0)
    cv2.putText(background, "Action " + str(agent_action), (30, 60), font, 0.5, (0, 0, 0), 1, 0)
    cv2.putText(background, "Receiving Teacher " + teacher_name, (30, 120), font, 0.5, (0, 0, 0), 1, 0)
    try:
        cv2.putText(background, "Feedback: " + str(feedback), (30, 150), font, 0.5, (0, 0, 0), 1, 0)
        cv2.putText(background, "Agent pos: " + str(env.get_pos()), (30, 180), font, 0.5, (0, 0, 0), 1, 0)
        cv2.putText(background, "Waypoint: " + str(env.waypoint_controller.waypoints[0]), (30, 210), font, 0.5, (0, 0, 0), 1, 0)
        cv2.putText(background, "Target: " + str(env.get_target()), (30, 240), font, 0.5, (0, 0, 0), 1, 0)
        cv2.putText(background, "All Waypoints: " + str(env.waypoint_controller.waypoints), (30, 270), font, 0.5, (0, 0, 0), 1, 0)
        if hasattr(env, 'teacher') and hasattr(env.teacher, 'teachers'):
            t = list(env.teacher.teachers.values())[0]
            cv2.putText(background, "Next Action: " + str(t.next_action), (30, 300), font, 0.5, (0, 0, 0), 1, 0)
        cv2.putText(background, "Reward: " + str(reward), (30, 330), font, 0.5, (0, 0, 0), 1, 0)
        cv2.putText(background, "Dist to goal: " + str(np.linalg.norm(env.get_target() - env.get_pos())), (30, 360), font, 0.5, (0, 0, 0), 1, 0)
    except Exception as e:
        pass
    return background


def rollout(env, agent, instr_dropout_prob=0, max_path_length=np.inf, speedup=1,
            video_directory="", video_name='sim_out', stochastic=False, num_rollouts=1,
            num_save=None, record_teacher=False, save_locally=True,
            save_wandb=False, teacher_name="", rollout_oracle=False,
            hierarchical_rollout=False):
    codec = 'MJPG'
    extension = '.avi'
    discrete = type(env.action_space) is Discrete
    video_filename = os.path.join(video_directory, video_name + extension)
    if num_save is None:
        num_save = num_rollouts

    # Get setup to log
    timestep = env.get_timestep()
    fps = int(speedup / timestep)

    success_writer = None
    failure_writer = None
    all_writer = None
    if save_locally:
        img = env.render(mode='rgb_array')
        height, width, channels = img.shape
        size = (width, height)
        all_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*codec), fps, size)
    if save_wandb:
        all_videos, success_videos, failure_videos = [], [], []

    # Collect a few trajectories
    paths, agent_actions, teacher_actions = [], [], []
    correct, stoch_correct, det_correct, count, total_reward, success = 0, 0, 0, 0, 0, 0
    for i in range(num_rollouts):
        observations, actions, rewards, agent_infos, env_infos, curr_images = [], [], [], [], [], []
        path_length = 0

        o = env.reset()
        # Loop until the max_path_length or we hit done
        while path_length < max_path_length:
            if hierarchical_rollout:
                del o['OffsetWaypoint']
                del o['gave_OffsetWaypoint']
            past_o = o
            # Choose action
            if hierarchical_rollout:
                a, agent_info = agent.get_hierarchical_actions([o])
            else:
                a, agent_info = agent.get_actions([o], instr_dropout_prob=instr_dropout_prob)
            stoch_a = a
            det_a = agent_info['argmax_action']
            if discrete:
                a = a.item()
                correct = int(a == env.teacher_action.item())
            else:
                correct = True
            if not stochastic:
                a = det_a
            if (save_locally or save_wandb) and i < num_save:
                image = env.render(mode='rgb_array')

            # Step env
            if rollout_oracle:
                next_o, r, d, env_info = env.step(env.teacher_action)
            else:
                next_o, r, d, env_info = env.step(a)

            # Store data for logging
            if discrete:
                if env_info['teacher_action'].item() == a:
                    correct += 1
                if env_info['teacher_action'].item() == stoch_a:
                    stoch_correct += 1
                if env_info['teacher_action'].item() == det_a:
                    det_correct += 1
            count += 1
            total_reward += r
            rewards.append(r)
            actions.append(a)
            env_infos.append(env_info)
            path_length += 1

            o = next_o

            # Render image, if necessary
            if (save_locally or save_wandb) and i < num_save:
                img = plot_img_intermediate(env, obs=past_o, image=image, agent_action=a, env_info=env_info,
                               record_teacher=record_teacher, teacher_name=teacher_name,
                               reward=r)
                curr_images.append(img)

            # End trajectory on 'done'
            if d:
                if env_info['success']:
                    success += 1
                print("Done on timestep", path_length, env_info['success'])
                break

        # At the end of a trajectory, save it
        if save_locally and i < num_save:
            write_traj_local(video_filename, curr_images, success, success_writer, failure_writer,
                    all_writer, fps, size, codec)
        if save_wandb and i < num_save:
            sample_img = np.zeros_like(curr_images[-1])
            all_videos += curr_images
            if success:
                success_videos += curr_images + [sample_img + 255] * 3
            else:
                failure_videos += curr_images + [sample_img + 255] * 3

        paths.append(dict(
            actions=actions,
            rewards=rewards,
            env_infos=env_infos
        ))

    # Finish saving videos
    if save_locally:
        finalize_videos_local(video_filename, all_writer, success_writer, failure_writer)
    if save_wandb:
        finalize_videos_wandb(video_name, all_videos, success_videos, failure_videos, fps)

    print(f"Finished rollouts, acc = {stoch_correct / count}, success = {success / num_rollouts}",)
    return paths, correct / count, stoch_correct / count, det_correct / count, total_reward / count
