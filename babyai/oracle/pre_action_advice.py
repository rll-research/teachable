import numpy as np
from babyai.oracle.teacher import Teacher



class PreActionAdvice(Teacher):

    def __init__(self, botclass, env, device=None, feedback_type='oracle', feedback_always=False):
        super().__init__(botclass, env, device, feedback_type, feedback_always)
        self.null_action = env.action_space.n

    def empty_feedback(self, env):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([self.null_action])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([self.null_action])

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        return np.array([self.next_action])

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        """
        return True



