import numpy as np
import pickle as pkl
from babyai.oracle.off_sparse_random_easy import OSREasy


class OSRMistaken(OSREasy):
    def __init__(self, *args, **kwargs):
        self.feedback_active = False
        super(OSRMistaken, self).__init__(*args, **kwargs)

    def compute_feedback(self, oracle, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        # Copy so we don't mess up the state of the real oracle
        oracle_copy = pkl.loads(pkl.dumps(oracle))
        self.step_ahead(oracle_copy, last_action=last_action)
        env = oracle.mission
        feedback = self.generic_feedback(env, offset=self.feedback_active)
        return np.concatenate([[int(self.feedback_active)], feedback])

    def empty_feedback(self, env=None):
        """
        Return a tensor corresponding to no feedback.
        """
        feedback = self.generic_feedback(env)
        return np.concatenate([[int(self.feedback_active)], feedback])

    def feedback_condition(self, oracle, action):
        """
        Returns true when we should give feedback, which happens every time the agent messes up
        """
        env = oracle.mission
        # Determines whether the current feedback is still relevant
        if (self.steps_since_lastfeedback % self.num_steps == 0) or np.array_equal(env.agent_pos, self.goal_coords):
            self.feedback_active = False
            self.steps_since_lastfeedback = self.feedback_frequency - 1
        if not self.last_action == action:
            self.feedback_active = True
            self.steps_since_lastfeedback = 0
            return True
        else:  # took the correct action
            return False

    def give_feedback(self, state, last_action, oracle):
        """
        Augment the agent's state observation with teacher feedback.
        :param oracle:
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        oracle = pkl.loads(pkl.dumps(oracle))
        feedback, gave_feedback = super().give_feedback(state, last_action, oracle)
        return feedback, gave_feedback
