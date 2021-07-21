import numpy as np
import pickle as pkl
from envs.babyai.oracle.teacher import Teacher


class OffsetCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(OffsetCorrections, self).__init__(*args, **kwargs)
        self.next_state_coords = self.empty_feedback()

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.zeros(5) - 1

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=5)

    def compute_feedback(self, oracle, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        # Copy so we don't mess up the state of the real oracle
        oracle_copy = pkl.loads(pkl.dumps(oracle))
        self.step_ahead(oracle_copy, last_action=last_action)
        return np.array(self.next_state_coords)

    # TODO: THIS IS NO IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True

    def step_ahead(self, oracle, last_action=-1):
        env = oracle.mission
        # Remove teacher so we don't end up with a recursion error
        env.teacher = None
        try:
            original_coords = np.concatenate([env.agent_pos, [env.agent_dir, int(env.carrying is not None)]])
            self.next_state, next_coords, _, _ = self.step_away_state(oracle, self.cartesian_steps,
                                                                      last_action=last_action)
            self.next_state_coords = np.concatenate([next_coords, [env.agent_dir]])
            self.next_state_coords[:4] -= original_coords
            # When we rotate, make sure it's always +/- 1
            if self.next_state_coords[2] == 3:
                self.next_state_coords[2] = -1
            elif self.next_state_coords[2] == -3:
                self.next_state_coords[2] = 1
            self.next_state_coords = self.next_state_coords.astype(np.float32)
        except Exception as e:
            print("STEP AWAY FAILED Offset!", e)
            self.next_state = self.next_state * 0
            self.next_state_coords = self.empty_feedback()
            self.last_step_error = True
        return oracle
