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
        return self.generic_feedback(env, offset=self.feedback_active)

    def feedback_condition(self, env, action=None):
        """
        Returns true when we should give feedback, which happens every time the agent messes up
        """
        # If we achieved our goal or have spent long enough chasing this one, pick a new one. We may or may not show it.
        achieved_goal = np.array_equal(env.agent_pos, self.goal_coords)
        timed_out = self.steps_since_lastfeedback % self.num_steps == 0
        if achieved_goal or timed_out:
            self.steps_since_lastfeedback = 0
            give_feedback = np.random.uniform() < .5
            self.feedback_active = give_feedback
            return give_feedback
        return False
