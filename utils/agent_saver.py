import pickle
import os


def save_object(entry, filename: str, prefix: str = "",
                folder="/Users/1000ber-5078/PycharmProjects/teachable-rl/test/"):
    if folder is not None and not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename_on_disk(filename, folder, prefix), 'wb') as f:
        pickle.dump(entry, f)
        f.close()


def load_object(filename: str, folder, prefix: str = ""):
    with open(filename_on_disk(filename, folder, prefix), 'rb') as f:
        obj = pickle.load(f)
        f.close()
        return obj


def filename_on_disk(filename: str, folder: str, prefix="") -> str:
    if len(prefix) > 0:
        prefix = prefix + "-"
    return f'{folder}/{prefix}{filename}.pickle'


def save_ppo_agent(agent, itr: str, folder: str = None) -> None:
    save_object(agent.args, "args", itr, folder)
    save_object(agent.teacher, "teacher", itr, folder)
    save_object(agent.env, "env", itr, folder)
    save_object(agent.critic, "critic", itr, folder)
    save_object(agent.actor, "actor", itr, folder)
    save_object(agent.advice_embedding, "advice_embedding", itr, folder)
    save_object(agent.reconstructor, "reconstructor", itr, folder)


# obs, batch, advice=None, no_advice_obs=None, next_obs=None
def save_ppo_agent_iteration(agent, obs, batch, advice, no_advice_obs, next_obs, itr: str) -> None:
    save_ppo_agent(agent, itr)

    save_object(agent.args, "args", itr)
    save_object(agent.env, "env", itr)
    save_object(agent.teacher, "teacher", itr)
    save_object(obs, "obs", itr)
    save_object(batch, "batch", itr)
    save_object(advice, "advice", itr)
    save_object(no_advice_obs, "no_advice_obs", itr)
    save_object(next_obs, "next_obs", itr)
