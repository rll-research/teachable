import numpy as np
from babyai.oracle.teacher import Teacher

class CartesianCorrections(Teacher):

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([-1, -1])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        hrand = np.random.randint(self.env.grid.height)
        wrand = np.random.randint(self.env.grid.width)
        return np.array([hrand, wrand])

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        # TODO: Unhardocde this
        # Hardcoded 1 time-step away

        if self.path is None:
            raise Exception('Path not found,blah')
        else:
            if len(self.path) > 1:
                feedback = self.path[1]
            elif len(self.path) == 0:
                feedback = self.env.obj_pos
            elif len(self.path) == 1:
                feedback = self.path[0]
        return np.array(feedback)

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.

        return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])



