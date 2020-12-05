import numpy as np
from babyai.oracle.teacher import Teacher

class SubgoalCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(SubgoalCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return -1*np.ones(17)

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=17)

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        return np.array(self.next_subgoal)

    def success_check(self, action):
        # subgoal_val = self.subgoal_to_idx(state)
        # followed_opt_action = np.allclose(subgoal_val, self.last_feedback)
        # return followed_opt_action
        opt_action = int(self.next_action)
        followed_opt_action = (opt_action == action[0])
        return followed_opt_action

    def give_feedback(self, state):
        """
        Augment the agent's state observation with teacher feedback.
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        feedback = self.compute_feedback()
        gave_feedback = self.last_feedback is None or not np.array_equal(feedback, self.last_feedback)
        self.last_feedback = feedback
        return feedback, gave_feedback
