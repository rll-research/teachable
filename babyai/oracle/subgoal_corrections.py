import numpy as np
from babyai.oracle.teacher import Teacher

class SubgoalCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(SubgoalCorrections, self).__init__(feedback_frequency=20, *args, **kwargs)

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
        # TODO: Unhardocde this
        # Hardcoded 1 time-step away
        # self.env_states, self.env_rewards, self.agent_positions = self.compute_full_path(1)
        # if len(self.env_states) > 0:
        #     feedback = self.env_states[0]
        # else:
        #     feedback = -1*np.ones(self.obs_size)
        # return np.array(feedback)
        return np.array(self.next_subgoal)
        
    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.
        if len(self.agent_actions) > 0 and (self.steps_since_lastfeedback % self.feedback_frequency == 0):
            self.steps_since_lastfeedback = 0
            return True
        else:
            return False
        # Old condition
        # return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])

    def success_check(self, action):
        # subgoal_val = self.subgoal_to_idx(state)
        # followed_opt_action = np.allclose(subgoal_val, self.last_feedback)
        # return followed_opt_action
        opt_action = int(self.next_action)
        followed_opt_action = (opt_action == action[0])
        return followed_opt_action