import numpy as np
import time
import cv2


def write_video(writer, frames, show_last=None):
    if show_last is not None:
        frames = frames[-show_last:]
    for frame in frames:
        writer.write(frame)

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, save_video=True, reset_every=1, batch_size=1,
            video_filename='sim_out.mp4', ignore_done=False, stochastic=False, num_rollouts=1, show_last=None,
            num_save=None, record_teacher=False):
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
    size = (width, height + 100)
    fps = int(speedup / timestep)
    if save_video:
        all_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        success_writer = None
        failure_writer = None


    render = animated or save_video
    for i in range(num_rollouts):
        # print("Rollout", i)
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        curr_images = []
        if i % reset_every == 0:
            agent.reset(dones=[True] * batch_size)
            env.set_task(None)
        path_length = 0
        o = env.reset()
        if render:
            _ = env.render(mode)
        while path_length < max_path_length:

            obs_big = np.stack([o] * batch_size)
            a, agent_info = agent.get_actions(obs_big)
            a = a[0]
            # print("advice", np.argmax(o[160:167]))
            if np.argmax(o[160:167]) == 7:
                print("No advice!")
            # a = np.array([[np.argmax(o[160:167])]])
            # a = np.array([[6]])
            if not stochastic:
                a = agent_info['mean']
            next_o, r, d, env_info = env.step(a)
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

            if save_video and i < num_save:
                image = env.render(mode='rgb_array')[:, :, ::-1] # RGB --> BGR
                title_block = np.zeros((100, image.shape[1], 3), np.uint8) + 255
                if not env_info['teacher_action'] == a[0][0]:
                    title_block[:, :, 0] = 0
                image = cv2.vconcat((title_block, image))
                font = cv2.FONT_HERSHEY_SIMPLEX
                label_str = ""
                if hasattr(env, "teacher") and env.teacher is not None:
                    if record_teacher:
                        label_str += f"Teacher advice: {env_info['teacher_action']}"
                    else:
                        label_str += f"Teacher on: {env.teacher.feedback_type}   "
                label_str += "Run: " + str(i % reset_every)
                cv2.putText(image, env.mission, (30, 30), font, 0.5, (0, 0, 0), 3, 0)
                cv2.putText(image, label_str, (30, 90), font, 0.5, (0, 0, 0), 3, 0)
                cv2.putText(image, "Action " + str(a[0][0]), (30, 60), font, 0.5, (0, 0, 0), 3, 0)

                curr_images.append(image)

            if d and not ignore_done:
                break

        if save_video and i < num_save:
            # Add a few blank frames at the end to indicate success (white) or failure (black)
            sample_img = np.zeros_like(curr_images[-1])
            if r > 0:
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

        paths.append(dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                agent_infos=agent_infos,
                env_infos=env_infos
            ))
    if save_video:
        all_writer.release()
        if success_writer is not None:
            success_writer.release()
        if failure_writer is not None:
            failure_writer.release()
        print("Video saved at %s" % video_filename)


    end = time.time()
    print("total time spent on rollouts", end - start)
    return paths
