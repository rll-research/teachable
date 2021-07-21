import numpy as np
from envs.babyai.oracle.teacher import Teacher

class PhysicalCorrections(Teacher):

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        hrand = np.random.randint(self.env.grid.height)
        wrand = np.random.randint(self.env.grid.width)
        self.env.agent_pos = np.array([hrand, wrand])
        return np.array([])

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        # TODO: Unhardocde this
        # Hardcoded 1 time-step away
        # TODO: Check for off by one errors here. 
        # Put in a signal that it has been moved
        if self.path is None:
            raise Exception('Path not found, blah')
        else:
            if len(self.path) > 1:
                feedback = self.path[1]
                self.env.agent_pos = np.array(feedback)
        return np.array([])

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.

        return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])