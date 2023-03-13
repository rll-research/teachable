import pickle

def save_object(entry, filename: str, prefix: str = ""):
    with open(filename_on_disk(filename, prefix), 'wb') as f:
        pickle.dump(entry, f)
        f.close()


def load_object(filename: str, prefix: str = ""):
    with open(filename_on_disk(filename, prefix), 'rb') as f:
        obj = pickle.load(f)
        f.close()
        return obj


def filename_on_disk(filename: str, prefix="") -> str:
    if len(prefix) > 0:
        prefix = prefix + "-"
    return f'/Users/1000ber-5078/PycharmProjects/teachable-rl/test/{prefix}{filename}.pickle'


# obs, batch, advice=None, no_advice_obs=None, next_obs=None
def save_ppo_agent_iteration(agent, obs, batch, advice, no_advice_obs, next_obs, itr: str) -> None:

    save_object(agent.critic, "critic", itr)
    save_object(agent.actor, "actor", itr)
    save_object(agent.advice_embedding, "advice_embedding", itr)
    save_object(agent.reconstructor, "reconstructor", itr)

    save_object(agent.args, "args", itr)
    save_object(agent.env, "env", itr)
    save_object(agent.teacher, "teacher", itr)
    save_object(obs,"obs",itr)
    save_object(batch,"batch",itr)
    save_object(advice,"advice",itr)
    save_object(no_advice_obs,"no_advice_obs",itr)
    save_object(next_obs,"next_obs",itr)

