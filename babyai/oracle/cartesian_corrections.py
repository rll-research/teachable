import numpy as np
from babyai.oracle.teacher import Teacher

class CartesianCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(CartesianCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return -1*np.ones(self.obs_size)

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=self.obs_size)

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        return np.array(self.next_state)
        


    def success_check(self, state):
        if self.last_feedback is None:
            return False
        followed_opt_action = np.allclose(state, self.last_feedback)
        return followed_opt_action

        # TODO: Check that success check works, no off by one error, and the feedback seems reasonable. 