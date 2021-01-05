import numpy as np
import pickle as pkl
from babyai.oracle.teacher import Teacher


class OffsetCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(OffsetCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.zeros(4) - 1

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=4)

    def compute_feedback(self, oracle):
        """
        Return the expert action from the previous timestep.
        """
        self.step_ahead(oracle)
        return np.array(self.next_state_coords)

    # TODO: THIS IS NO IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True

    def step_ahead(self, oracle):
        original_teacher = oracle.mission.teacher
        oracle.mission.teacher = None
        env = pkl.loads(pkl.dumps(oracle.mission))
        env.teacher = None
        try:
            original_coords = np.concatenate([env.agent_pos, [env.agent_dir, int(env.carrying is not None)]])
            self.next_state, next_coords = self.step_away_state(env, oracle, self.cartesian_steps)
            self.next_state_coords = next_coords - original_coords
        except Exception as e:
            print("STEP AWAY FAILED!", e)
            self.next_state = self.next_state * 0
            self.next_state_coords = np.zeros(4) - 1
            self.last_step_error = True
        oracle.mission.teacher = original_teacher
        return oracle
