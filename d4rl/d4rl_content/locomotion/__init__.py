from gym.envs.registration import register
from d4rl_content.locomotion import ant
from d4rl_content.locomotion import maze_env

"""
register(
    id='antmaze-umaze-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)
"""

register(
    id='antmaze-open-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=350,
    kwargs={
        'maze_map': maze_env.OPEN,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-easy-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=350,
    kwargs={
        'maze_map': maze_env.U_MAZE_CLOSEGOAL,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)


register(
    id='antmaze-randommaze-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=800,
    kwargs={
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'maze_size': 6,
    }
)

register(
    id='antmaze-randommaze-medium-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1200,
    kwargs={
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'maze_size': 5,
    }
)

for i in range(10):
    register(
        id=f'antmaze-fixed{i}-6x6-v0',
        entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
        max_episode_steps=500,
        kwargs={
            'maze_map': getattr(maze_env, f'M{i}'),
            'reward_type': 'sparse',
            'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
            'non_zero_reset': False,
            'eval': True,
            'ref_min_score': 0.0,
            'ref_max_score': 1.0,
            'maze_size': 5,
        }
    )


register(
    id='antmaze-randommaze-small-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=300,
    kwargs={
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'maze_size': 4,
    }
)

register(
    id='antmaze-randommaze-large-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=2000,
    kwargs={
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'maze_size': 8,
    }
)

register(
    id='antmaze-randommaze-huge-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=2000,
    kwargs={
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'maze_size': 10,
    }
)

register(
    id='antmaze-6x6-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=800,
    kwargs={
        'maze_map': maze_env.STATIC_6_BY_6,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)


register(
    id='antmaze-umaze-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1200,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-diverse-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-play-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-diverse-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=2000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-diverse-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1400,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-play-v0',
    entry_point='d4rl_content.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)
