import numpy as np
from babyai.oracle.teacher import Teacher


class DummyAdvice(Teacher):
    def __init__(self, botclass, env, *args, **kwargs):
        super(DummyAdvice, self).__init__(botclass, env, *args, **kwargs)
        self.next_action = env.action_space.sample() * 0 - 1

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([])

    def compute_feedback(self, *args, **kwargs):
        """
        Return the expert action from the previous timestep.
        """
        return np.array([])

    def success_check(self, *args, **kwargs):
        return False

    def reset(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass