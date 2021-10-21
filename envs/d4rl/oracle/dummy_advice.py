import numpy as np
from envs.d4rl.oracle.teacher import Teacher


class DummyAdvice(Teacher):
    def __init__(self, *args, **kwargs):
        super(DummyAdvice, self).__init__(*args, **kwargs)
        # self.next_action = None

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

    def feedback_condition(self):
        return False