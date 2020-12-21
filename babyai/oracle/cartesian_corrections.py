import numpy as np
import pickle as pkl
from babyai.oracle.teacher import Teacher

class CartesianCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(CartesianCorrections, self).__init__(*args, **kwargs)

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

    def compute_feedback(self, oracle):
        """
        Return the expert action from the previous timestep.
        """
        self.step_ahead(oracle)
        return np.array(self.next_state)

    def success_check(self, state, action, oracle):
        if self.last_feedback is None:
            return False
        followed_opt_action = np.allclose(state.flatten(), self.past_timestep_feedback.flatten())
        return followed_opt_action

    def step_ahead(self, oracle):
        original_teacher = oracle.mission.teacher
        oracle.mission.teacher = None
        env_copy1 = pkl.loads(pkl.dumps(oracle.mission))
        env_copy1.teacher = None
        try:
            self.next_state = self.step_away_state(env_copy1, oracle, self.cartesian_steps)
        except Exception as e:
            print("STEP AWAY FAILED!")
            print(e)
            print("CURRENT VISMASK", oracle.vis_mask)
            self.next_state = self.next_state * 0
            self.last_step_error = True
        oracle.mission.teacher = original_teacher
        return oracle
