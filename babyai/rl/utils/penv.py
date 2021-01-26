from multiprocessing import Process, Pipe
import gym

def worker(conn, env, rollouts_per_meta_task, seed):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                if env.itr == rollouts_per_meta_task:
                    if seed is not None:
                        env.seed(seed)
                    env.set_task(None)
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            if seed is not None:
                env.seed(seed)
            env.set_task(None)
            obs = env.reset()
            conn.send(obs)
        elif cmd == 'advance_curriculum':
            result = env.advance_curriculum()
            conn.send(result)
        elif cmd == "set_task":
            obs = env.set_task(None)
            conn.send(obs)
        elif cmd == "render":
            obs = env.render(mode='rgb_array')
            conn.send(obs)
        elif cmd == "get_teacher_action":
            result = env.get_teacher_action()
            conn.send(result)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, rollouts_per_meta_task, repeated_seed=None):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.locals = []
        self.rollouts_per_meta_task = rollouts_per_meta_task
        self.repeated_seed = repeated_seed
        self.reset_processes()

    def reset_processes(self):
        if len(self.locals) > 0:
            self.end_processes()
        self.locals = []
        self.processes = []
        repeated_seed = self.repeated_seed if self.repeated_seed is not None else [None] * len(self.envs)
        for env, seed in zip(self.envs[1:], repeated_seed[1:]):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, self.rollouts_per_meta_task, seed))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        if self.repeated_seed is not None:
            self.envs[0].seed(self.repeated_seed[0])
        self.envs[0].set_task(None)
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def advance_curriculum(self):
        for local in self.locals:
            local.send(("advance_curriculum", None))
        results = [self.envs[0].advance_curriculum()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))  # TODO: does this reset?
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            if self.repeated_seed is not None:
                self.envs[0].seed(self.repeated_seed[0])
            self.envs[0].set_task(None)
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        results = list(results)
        return results

    def update_tasks(self):
        for env in self.envs:
            env.set_task(None)
        self.reset_processes()

    def render(self):
        for local in self.locals:
            local.send(("render", None))
        results = [self.envs[0].render(mode='rgb_array')] + [local.recv() for local in self.locals]
        return results

    def get_teacher_action(self):
        for local in self.locals:
            local.send(("get_teacher_action", None))
        results = [self.envs[0].get_teacher_action()] + [local.recv() for local in self.locals]
        return results

    def __del__(self):
        self.end_processes()

    def end_processes(self):
        for p in self.processes:
            p.terminate()


class SequentialEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, rollouts_per_meta_task, repeated_seed=None):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.locals = []
        self.rollouts_per_meta_task = rollouts_per_meta_task
        self.repeated_seed = repeated_seed
        assert repeated_seed is None or len(repeated_seed) == len(envs), \
            f"repeated seed has length {len(repeated_seed)} but should have length {len(envs)}"

    def reset(self):
        results = []
        for i, env in enumerate(self.envs):
            if self.repeated_seed is not None:
                env.seed(self.repeated_seed[i])
            env.set_task(None)
            results.append(env.reset())
        return results

    def advance_curriculum(self):
        results = []
        for env in self.envs:
            results.append(env.advance_curriculum())
        return results

    def step(self, actions):
        results = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(action)
            if done:
                if env.itr == self.rollouts_per_meta_task:
                    if self.repeated_seed is not None:
                        env.seed(self.repeated_seed[i])
                    env.set_task(None)
                obs = env.reset()
            results.append((obs, reward, done, info))
        results = zip(*results)
        return results

    def update_tasks(self):
        for env in self.envs:
            env.set_task(None)

    def render(self):
        results = [env.render(mode='rgb_array') for env in self.envs]
        return results

    def get_teacher_action(self):
        results = [env.get_teacher_action() for env in self.envs]
        return results