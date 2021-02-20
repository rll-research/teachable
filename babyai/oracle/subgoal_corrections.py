import numpy as np
from babyai.oracle.teacher import Teacher


class SubgoalCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(SubgoalCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return -1 * np.ones(18)

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=18)

    def compute_feedback(self, _, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        return np.array(self.next_subgoal)

    def success_check(self, state, action, oracle):
        """ Assume the agent completed the subgoal when the last subgoal is no longer in the stack. """
        stack = oracle.stack
        for subgoal in stack:
            if np.allclose(self.last_feedback, oracle.subgoal_to_index(subgoal)):
                return False
        return True

    def give_feedback(self, state, action, oracle):
        """
        Augment the agent's state observation with teacher feedback.
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        feedback = self.compute_feedback(None)
        gave_feedback = self.last_feedback is None or not np.array_equal(feedback[:19], self.last_feedback[:19])
        self.last_feedback = feedback
        return feedback, gave_feedback
