import numpy as np
from babyai.oracle.teacher import Teacher


class DummyAdvice(Teacher):
    def __init__(self, *args, **kwargs):
        super(DummyAdvice, self).__init__(*args, **kwargs)

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

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        return np.array([])

    def success_check(self, action, _, another):
        opt_action = int(self.next_action)
        followed_opt_action = (opt_action == action)
        return followed_opt_action
