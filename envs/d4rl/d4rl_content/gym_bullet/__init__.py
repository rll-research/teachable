from gym.envs.registration import register
from envs.d4rl.d4rl_content.gym_bullet import gym_envs
from envs.d4rl.d4rl_content import infos
import gym

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'bullet' in env:
        print('Remove {} from registry'.format(env))
        del gym.envs.registration.registry.env_specs[env]

for agent in ['hopper', 'halfcheetah', 'ant', 'walker2d']:
    register(
        id='bullet-%s-v0' % agent,
        entry_point='d4rl_content.gym_bullet.gym_envs:get_%s_env' % agent,
        max_episode_steps=1000,
    )

    for dataset in ['random', 'medium', 'expert', 'medium-expert', 'medium-replay']:
        env_name = 'bullet-%s-%s-v0' % (agent, dataset)
        register(
            id=env_name,
            entry_point='d4rl_content.gym_bullet.gym_envs:get_%s_env' % agent,
            max_episode_steps=1000,
            kwargs={
                'ref_min_score': infos.REF_MIN_SCORE[env_name],
                'ref_max_score': infos.REF_MAX_SCORE[env_name],
                'dataset_url': infos.DATASET_URLS[env_name]
            }
        )

