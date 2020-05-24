import numpy as np
import time
import moviepy.editor as mpy
from matplotlib import pyplot as plt


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, save_video=True, reset_every=1, batch_size=1, save_failures=True,
            video_filename='sim_out.mp4', ignore_done=False, stochastic=False, num_rollouts=1, show_last=None):
    start = time.time()
    if hasattr(env, 'dt'):
        timestep = env.dt
    else:
        timestep = 0.2
    images = []
    paths = []
    if animated:
        mode = 'human'
    else:
        mode = 'rgb_array'

    render = animated or save_video
    fig = plt.figure()
    for i in range(num_rollouts):
        print("Rollout", i)
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        curr_images = []
        failures = []
        successes = []
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

            if save_video:
                image = env.render(mode='rgb_array')

                plt.imshow(image)
                plt.title(env.mission)
                label_str = ""
                if hasattr(env, "teacher"):
                    label_str += "Teacher on:    "
                label_str += "Run: " + str(i % reset_every)
                plt.xlabel(label_str)
                fig.canvas.draw()
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                fig.clf()

                curr_images.append(image)

            if d and not ignore_done:
                break

        if save_video:
            # Add a few blank frames at the end to indicate success (white) or failure (black)
            sample_img = np.zeros_like(curr_images[-1])
            if r > 0:
                curr_images += [sample_img + 255] * 3
                successes += curr_images
            else:
                curr_images += [sample_img] * 3
                failures += curr_images

            # If show_last is enabled, only show the end of the trajectory.
            if not show_last:
                images += curr_images
            else:
                images += curr_images[-show_last - 3:]

        paths.append(dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                agent_infos=agent_infos,
                env_infos=env_infos
            ))
    if save_video:
        fps = int(speedup/timestep)
        clip = mpy.ImageSequenceClip(images, fps=fps)
        if video_filename[-3:] == 'gif':
            clip.write_gif(video_filename, fps=fps)
        else:
            clip.write_videofile(video_filename, fps=fps)
        print("Video saved at %s" % video_filename)

        if len(failures) > 0:
            clip = mpy.ImageSequenceClip(failures, fps=fps)
            if video_filename[-3:] == 'gif':
                clip.write_gif(video_filename[:-4] + "failures" + ".gif", fps=fps)
            else:
                clip.write_videofile(video_filename[:-4] + "failures" + video_filename[-4:], fps=fps)
        if len(successes) > 0:
            clip = mpy.ImageSequenceClip(successes, fps=fps)
            if video_filename[-3:] == 'gif':
                clip.write_gif(video_filename[:-4] + "successes" + ".gif", fps=fps)
            else:
                clip.write_videofile(video_filename[:-4] + "successes" + video_filename[-4:], fps=fps)

    plt.close(fig)
    end = time.time()
    print("total time spent on rollouts", end - start)
    return paths
