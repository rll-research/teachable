import numpy as np
from babyai.oracle.teacher import Teacher


class PreActionAdviceBad1(Teacher):
    def __init__(self, *args, **kwargs):
        super(PreActionAdviceBad1, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.concatenate([self.one_hotify(-1), self.one_hotify(-1), self.one_hotify(-1),
                              self.get_last_feedback_indicator()])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return self.one_hotify(self.action_space.sample())

    def compute_feedback(self, _, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        random = np.random.randint(8, size=3)
        true_index = np.random.randint(3)
        arrs = [self.one_hotify([i]) for i in random]
        arrs[true_index] = self.one_hotify(self.next_action)
        return np.concatenate(arrs + [true_index])

    def one_hotify(self, index):
        correction = np.zeros((self.action_space.n + 1,))
        correction[index] = 1.0
        return correction

    def success_check(self, state, action, oracle):
        opt_action = int(self.last_action)
        followed_opt_action = (opt_action == action)
        return followed_opt_action and self.gave_feedback
