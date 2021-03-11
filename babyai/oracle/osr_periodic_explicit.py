import numpy as np
import pickle as pkl
from babyai.oracle.off_sparse_random_easy import OSREasy


class OSRPeriodicExplicit(OSREasy):
    def __init__(self, *args, **kwargs):
        self.feedback_active = False
        super(OSRPeriodicExplicit, self).__init__(*args, **kwargs)

    def compute_feedback(self, oracle, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        # Copy so we don't mess up the state of the real oracle
        oracle_copy = pkl.loads(pkl.dumps(oracle))
        self.step_ahead(oracle_copy, last_action=last_action)
        env = oracle.mission
        self.generic_feedback(env, offset=self.feedback_active)

    def feedback_condition(self, env, action):
        """
        Returns true when we should give feedback, which happens every time the agent messes up
        """
        # Determines whether the current feedback is still relevantt
        if (self.steps_since_lastfeedback % self.num_steps == 0) or np.array_equal(env.agent_pos, self.goal_coords):
            self.steps_since_lastfeedback = 0
            # half of the time, give

            self.feedback_active = False
            self.steps_since_lastfeedback = self.feedback_frequency - 1
        if not self.last_action == action:
            self.feedback_active = True
            self.steps_since_lastfeedback = 0
            return True
        else:  # took the correct action
            return False
