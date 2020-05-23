import numpy as np
from babyai.oracle.teacher import Teacher



class PreActionAdvice(Teacher):

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

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.
        return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])



