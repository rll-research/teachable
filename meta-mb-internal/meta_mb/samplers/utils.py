import numpy as np
import time
import cv2
# import wandb
import os

def write_video(writer, frames, show_last=None):
    if show_last is not None:
        frames = frames[-show_last:]
    for frame in frames:
        writer.write(frame)

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, reset_every=1, batch_size=1,
            video_directory="", video_name='sim_out', ignore_done=False, stochastic=False, num_rollouts=1, show_last=None,
            num_save=None, record_teacher=False, reward_predictor=None, use_teacher=False, save_locally=True,
            save_wandb=False):
    video_filename = os.path.join(video_directory, video_name + ".mp4")
    if num_save is None:
        num_save = num_rollouts
    start = time.time()
    if hasattr(env, 'dt'):
        timestep = env.dt
    else:
        timestep = 0.5#0.1
    images = []
    paths = []
    if animated:
        mode = 'human'
    else:
        mode = 'rgb_array'

    img = env.render(mode='rgb_array')
    height, width, channels = img.shape
    size = (width * 2, height * 2)
    fps = int(speedup / timestep)
    if save_locally:
        all_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        success_writer = None
        failure_writer = None
    if save_wandb:
        all_videos = []
        success_videos = []
        failure_videos = []

    correct = 0
    count = 0
    agent_actions = []
    teacher_actions = []
    render = animated or save_locally or save_locally
    for i in range(1):
        # print("Rollout", i)
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        curr_images = []
        if i % reset_every == 0:
            agent.reset(dones=[True] * batch_size)
            if reward_predictor is not None:
                reward_predictor.reset(dones=[True] * batch_size)
            env.set_task(None)
        path_length = 0
        o = env.reset()
        if render:
            _ = env.render(mode)
        while path_length < max_path_length:
            obs_big = np.stack([o] * batch_size)
            a, agent_info = agent.get_actions(obs_big, use_teacher=True)
            a = a[0][0]
            agent_actions.append(a)

            count += 1

            # if not stochastic:
            a = np.array([np.argmax(agent_info[0][0]['probs'])])
            next_o, r, d, env_info = env.step(a)
            success = env_info['success']

            teacher_actions.append(env_info['teacher_action'])
            if env_info['teacher_action'] == a:
                correct += 1

            if reward_predictor is not None:
                reward_obs = np.stack([env_info['next_obs_rewardfree']] * batch_size)
                pred_reward = reward_predictor.get_actions(reward_obs, use_teacher=True)


            observations.append(o)
            rewards.append(r)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1

            o = next_o

            if animated:
                env.render(mode)
                time.sleep(timestep/speedup)

            if (save_locally or save_wandb) and i < num_save:
                image = env.render(mode='rgb_array')[:, :, ::-1] # RGB --> BGR
                h, w, c = image.shape
                background = np.zeros((h * 2, w * 2, c), dtype=np.uint8) + 255

                # If we have a reward predictor, the background color is based on whether the reward predictor is correct
                if reward_predictor is not None:
                    pred_reward = np.round(pred_reward[0][0][0][0])
                    if pred_reward == 0 and r == 1:
                        background[:, :, 1] = 0
                    elif pred_reward == 1 and r == 0:
                        background[:, :, 2] = 0
                # Otherwise, the background color is based on whether the action taken is correct
                else:
                    if not env_info['teacher_action'] == a:
                        background[:, :, 0] = 0
                background[h:, w:] = image
                font = cv2.FONT_HERSHEY_SIMPLEX
                label_str = ""
                if hasattr(env, "teacher") and env.teacher is not None:
                    if record_teacher:
                        label_str += f"Teacher advice: {env_info['teacher_action']}"
                    else:
                        label_str += f"Teacher on: {env.teacher.feedback_type}   "
                label_str += "Run: " + str(i % reset_every)
                cv2.putText(background, env.mission, (30, 30), font, 0.5, (0, 0, 0), 3, 0)
                cv2.putText(background, label_str, (30, 90), font, 0.5, (0, 0, 0), 3, 0)
                cv2.putText(background, "Action " + str(a[0]), (30, 60), font, 0.5, (0, 0, 0), 3, 0)

                curr_images.append(background)

            if d and not ignore_done:
                break

        if save_locally and i < num_save:
            # Add a few blank frames at the end to indicate success (white) or failure (black)
            sample_img = np.zeros_like(curr_images[-1])
            if success:
                curr_images += [sample_img + 255] * 3
                if success_writer is None:
                    success_writer = cv2.VideoWriter(video_filename[:-4] + "success" + video_filename[-4:], cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                write_video(success_writer, curr_images)
            else:
                curr_images += [sample_img] * 3
                if failure_writer is None:
                    failure_writer = cv2.VideoWriter(video_filename[:-4] + "failures" + video_filename[-4:], cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                write_video(failure_writer, curr_images)
            write_video(all_writer, curr_images, show_last)
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
    if save_locally:
        all_writer.release()
        if success_writer is not None:
            success_writer.release()
        if failure_writer is not None:
            failure_writer.release()
        print("Video saved at %s" % video_filename)
    if save_wandb:
        video = np.transpose(np.stack(all_videos), (0, 3, 1, 2))[:, ::-1]
        # wandb.log({video_name + '_all': wandb.Video(video, fps=fps, format="mp4")}, commit=False)
        # if len(success_videos) > 0:
        #     video = np.transpose(np.stack(success_videos), (0, 3, 1, 2))[:, ::-1]
        #     wandb.log({video_name + '_success': wandb.Video(video, fps=fps, format="mp4")}, commit=False)
        # if len(failure_videos) > 0:
        #     video = np.transpose(np.stack(failure_videos), (0, 3, 1, 2))[:, ::-1]
        #     wandb.log({video_name + '_failure': wandb.Video(video, fps=fps, format="mp4")}, commit=False)

    end = time.time()
    print("total time spent on rollouts", end - start)
    print("ACCURACY", correct / count)
    print('Average Success Rate: ', np.mean([path['env_infos'][-1]['success'] for path in paths]))
    print("ROLLOUT ACTION COUNTS", np.unique(agent_actions, return_counts=True))
    print("ROLLOUT TEACHER COUNTS", np.unique(teacher_actions, return_counts=True))
    return paths, correct / count
