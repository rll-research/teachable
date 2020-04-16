from .acrobot import AcrobotEnv
from .ant import AntEnv
from .cartpole import CartPoleEnv
from .half_cheetah import HalfCheetahEnv
from .hopper import HopperEnv
from .inverted_pendulum import InvertedPendulumEnv
from .mountain_car import Continuous_MountainCarEnv
from .pendulum import PendulumEnv
from .reacher import ReacherEnv
from .swimmer import SwimmerEnv
from .walker2d import Walker2dEnv
from .slimhumanoid import SlimHumanoidEnv
from .nostopslimhumanoid import NoStopSlimHumanoidEnv
from .pendulumO01 import PendulumO01Env
from .pendulumO001 import PendulumO001Env
from .half_cheetahA01 import HalfCheetahA01Env
from .half_cheetahA003 import HalfCheetahA003Env
from .half_cheetahO01 import HalfCheetahO01Env
from .half_cheetahO001 import HalfCheetahO001Env
from .fwalker2d import FWalker2dEnv
from .fswimmer import FSwimmerEnv
from .fhopper import FHopperEnv
from .fant import FAntEnv
from .cartpoleO01 import CartPoleO01Env
from .cartpoleO001 import CartPoleO001Env
from .humanoid import HumanoidEnv


from gym.envs.registration import register

register(
    id='MBRLHalfCheetah-v0',
    entry_point='envs.gym.half_cheetah:HalfCheetahEnv',
    kwargs={'frame_skip': 5},
    max_episode_steps=1000,
)

register(
    id='MBRLWalker2d-v0',
    entry_point='envs.gym.walker2d:Walker2dEnv',
    kwargs={'frame_skip': 4},
    max_episode_steps=1000,
)

register(
    id='MBRLSwimmer-v0',
    entry_point='envs.gym.swimmer:SwimmerEnv',
    kwargs={'frame_skip': 4},
    max_episode_steps=1000,
)

register(
    id='MBRLAnt-v0',
    entry_point='envs.gym.ant:AntEnv',
    kwargs={'frame_skip': 5},
    max_episode_steps=1000,
)

register(
    id='MBRLHopper-v0',
    entry_point='envs.gym.hopper:HopperEnv',
    kwargs={'frame_skip': 4},
    max_episode_steps=1000,
)

register(
    id='MBRLReacher-v0',
    entry_point='envs.gym.reacher:ReacherEnv',
    max_episode_steps=50,
)


# second batch of environments

register(
    id='MBRLInvertedPendulum-v0',
    entry_point='envs.gym.inverted_pendulum:InvertedPendulumEnv',
    max_episode_steps=100,
)
register(
    id='MBRLAcrobot-v0',
    entry_point='envs.gym.acrobot:AcrobotEnv',
    max_episode_steps=200,
)
register(
    id='MBRLCartpole-v0',
    entry_point='envs.gym.cartpole:CartPoleEnv',
    max_episode_steps=200,

)
register(
    id='MBRLMountain-v0',
    entry_point='envs.gym.mountain_car:Continuous_MountainCarEnv',
    max_episode_steps=200,

)
register(
    id='MBRLPendulum-v0',
    entry_point='envs.gym.pendulum:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='gym_petsPusher-v0',
    entry_point='envs.gym.pets_pusher:PusherEnv',
    max_episode_steps=150,
)
register(
    id='gym_petsReacher-v0',
    entry_point='envs.gym.pets_reacher:Reacher3DEnv',
    max_episode_steps=150,
)
register(
    id='gym_petsCheetah-v0',
    entry_point='envs.gym.pets_cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
)

# noisy env
register(
    id='gym_cheetahO01-v0',
    entry_point='envs.gym.gym_cheetahO01:HalfCheetahEnv',
    max_episode_steps=1000,
)
register(
    id='gym_cheetahO001-v0',
    entry_point='envs.gym.gym_cheetahO001:HalfCheetahEnv',
    max_episode_steps=1000,
)
register(
    id='gym_cheetahA01-v0',
    entry_point='envs.gym.gym_cheetahA01:HalfCheetahEnv',
    max_episode_steps=1000,
)
register(
    id='gym_cheetahA003-v0',
    entry_point='envs.gym.gym_cheetahA003:HalfCheetahEnv',
    max_episode_steps=1000,
)
register(
    id='gym_pendulumO01-v0',
    entry_point='envs.gym.gym_pendulumO01:PendulumEnv',
    max_episode_steps=200,
)
register(
    id='gym_pendulumO001-v0',
    entry_point='envs.gym.gym_pendulumO001:PendulumEnv',
    max_episode_steps=200,
)
register(
    id='gym_cartpoleO01-v0',
    entry_point='envs.gym.gym_cartpoleO01:CartPoleEnv',
    max_episode_steps=200,
)
register(
    id='gym_cartpoleO001-v0',
    entry_point='envs.gym.gym_cartpoleO001:CartPoleEnv',
    max_episode_steps=200,
)

env_name_to_gym_registry = {
    # first batch
    "half_cheetah": "MBRLHalfCheetah-v0",
    "swimmer": "MBRLSwimmer-v0",
    "ant": "MBRLAnt-v0",
    "hopper": "MBRLHopper-v0",
    "reacher": "MBRLReacher-v0",
    "walker2d": "MBRLWalker2d-v0",

    # second batch
    "invertedPendulum": "MBRLInvertedPendulum-v0",
    "acrobot": 'MBRLAcrobot-v0',
    "cartpole": 'MBRLCartpole-v0',
    "mountain": 'MBRLMountain-v0',
    "pendulum": 'MBRLPendulum-v0',

    # the pets env
    "gym_petsPusher": "gym_petsPusher-v0",
    "gym_petsReacher": "gym_petsReacher-v0",
    "gym_petsCheetah": "gym_petsCheetah-v0",

    # the noise env
    "gym_cheetahO01": "gym_cheetahO01-v0",
    "gym_cheetahO001": "gym_cheetahO001-v0",
    "gym_cheetahA01": "gym_cheetahA01-v0",
    "gym_cheetahA003": "gym_cheetahA003-v0",

    "gym_pendulumO01": "gym_pendulumO01-v0",
    "gym_pendulumO001": "gym_pendulumO001-v0",

    "gym_cartpoleO01": "gym_cartpoleO01-v0",
    "gym_cartpoleO001": "gym_cartpoleO001-v0",
}
