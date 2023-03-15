from envs.babyai.utils.obs_preprocessor import make_obs_preprocessor
from utils.agent_saver import *
from algos.ppo import *


def load_agent_iteration(itr: str, folder="test"):
    args = load_object("args", folder, itr)
    teacher = load_object("teacher", folder, itr)
    env = load_object("env", folder, itr)

    obs_preprocessor = make_obs_preprocessor(args.feedback_list)

    agent = PPOAgent(args=args, obs_preprocessor=obs_preprocessor, teacher=teacher, env=env)

    agent.critic = load_object("critic", folder, itr)
    agent.actor = load_object("actor", folder, itr)
    agent.advice_embedding = load_object("advice_embedding", folder, itr)
    agent.reconstructor = load_object("reconstructor", folder, itr)
    return agent


def load_update_actor_params(itr: str):
    obs = load_object("obs", itr)
    batch = load_object("batch", itr)
    advice = load_object("advice", itr)
    no_advice_obs = load_object("no_advice_obs", itr)
    next_obs = load_object("next_obs", itr)
    return obs, batch, advice, no_advice_obs, next_obs
