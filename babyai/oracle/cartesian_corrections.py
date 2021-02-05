import numpy as np
import pickle as pkl
from babyai.oracle.teacher import Teacher


class CartesianCorrections(Teacher):
    def __init__(self, *args, obs_size=None, **kwargs):
        self.obs_size = obs_size
        super(CartesianCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return -1 * np.ones(self.obs_size)

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=self.obs_size)

    def compute_feedback(self, oracle, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        # Copy so we don't mess up the state of the real oracle
        oracle_copy = pkl.loads(pkl.dumps(oracle))
        self.step_ahead(oracle_copy, last_action=last_action)
        return np.array(self.next_state).flatten()

    def success_check(self, state, action, oracle):
        if self.past_timestep_feedback is None:
            return False
        followed_opt_action = np.allclose(state.flatten(), self.past_timestep_feedback.flatten())
        return followed_opt_action

    def step_ahead(self, oracle, last_action=-1):
        # Remove teacher so we don't end up with a recursion error
        oracle.mission.teacher = None
        try:
            self.next_state, self.next_state_coords = self.step_away_state(oracle, self.cartesian_steps,
                                                                           last_action=last_action)
        except Exception as e:
            print("STEP AWAY FAILED CC!", e)
            self.next_state = self.next_state * 0
            self.last_step_error = True
        return oracle
