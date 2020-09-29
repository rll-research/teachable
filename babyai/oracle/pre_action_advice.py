import numpy as np
from babyai.oracle.teacher import Teacher



class PreActionAdvice(Teacher):
    def __init__(self, *args, **kwargs):
        super(PreActionAdvice, self).__init__(*args, **kwargs)
                        
    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([-1])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([self.env.action_space.sample()])

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        return np.array([self.next_action])

    def success_check(self, action):
        opt_action = int(self.next_action)
        followed_opt_action = (opt_action == action)
        return followed_opt_action