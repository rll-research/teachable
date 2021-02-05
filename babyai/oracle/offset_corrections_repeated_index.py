import numpy as np
import pickle as pkl
from babyai.oracle.offset_corrections import OffsetCorrections


class OFFIO(OffsetCorrections):
    def __init__(self, *args, **kwargs):
        self.next_state_coords = np.zeros(5) - 1
        super(OFFIO, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.concatenate([self.next_state_coords, self.get_last_feedback_indicator()])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.concatenate([np.random.uniform(0, 1, size=5), self.get_last_feedback_indicator()])

    def compute_feedback(self, oracle, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        # Copy so we don't mess up the state of the real oracle
        oracle_copy = pkl.loads(pkl.dumps(oracle))
        self.step_ahead(oracle_copy, last_action=last_action)
        return np.concatenate([self.next_state_coords, self.get_last_feedback_indicator()])