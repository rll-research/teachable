import numpy as np
from envs.d4rl.oracle.teacher import Teacher


class DirectionCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(DirectionCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.zeros(2) - 1

    def give_feedback(self, env):
        gave_feedback = True
        return self.next_action, gave_feedback
