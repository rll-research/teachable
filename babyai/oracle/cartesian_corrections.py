import numpy as np
from babyai.oracle.teacher import Teacher

class CartesianCorrections(Teacher):

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return -1*np.ones_like(self.env.gen_obs())

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=self.env.gen_obs().shape)

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        # TODO: Unhardocde this
        # Hardcoded 1 time-step away
        self.env_states, self.env_rewards, self.agent_positions = self.compute_path_forward(-1)
        if len(self.path) > 1:
            feedback = self.env_states[1]
        else:
            feedback = self.env_states[0]
        return np.array(feedback)

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.

        return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])



