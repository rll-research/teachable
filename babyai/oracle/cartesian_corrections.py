import numpy as np
from babyai.oracle.teacher import Teacher

class CartesianCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(CartesianCorrections, self).__init__(feedback_frequency=5, *args, **kwargs)

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
        # TODO: Unhardocde this
        # Hardcoded 1 time-step away
        # self.env_states, self.env_rewards, self.agent_positions = self.compute_full_path(1)
        # if len(self.env_states) > 0:
        #     feedback = self.env_states[0]
        # else:
        #     feedback = -1*np.ones(self.obs_size)
        # return np.array(feedback)
        return np.array(self.next_state)
        
    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.
        # TODO NOW Fix this
        if len(self.agent_actions) > 0 and (self.steps_since_lastfeedback % self.feedback_frequency == 0):
            self.steps_since_lastfeedback = 0
            return True
        else:
            return False

        # Old condition
        # return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])

    def success_check(self, state):
        if self.last_feedback is None:
            return False
        followed_opt_action = np.allclose(state, self.last_feedback)
        return followed_opt_action

        # TODO: Check that success check works, no off by one error, and the feedback seems reasonable. 