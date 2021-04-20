import numpy as np
from oracle.teacher import Teacher


class DirectionCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(DirectionCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.zeros(2) - 1

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=2)

    def give_feedback(self, env):
        gave_feedback = True
        return self.next_action, gave_feedback

    # TODO: THIS IS NOT IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True
