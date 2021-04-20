import numpy as np
from oracle.teacher import Teacher


class CardinalCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(CardinalCorrections, self).__init__(*args, **kwargs)

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

    def give_feedback(self, env):
        # Always re-compute feedback, but only count it as new feedback if it changes
        down_dist = self.next_action[0]
        up_dist = -down_dist
        right_dist = self.next_action[1]
        left_dist = -right_dist
        cardinal_dir_scalar = np.argmax([up_dist, right_dist, down_dist, left_dist])
        cardinal_dir_one_hot = np.zeros(4)
        cardinal_dir_one_hot[cardinal_dir_scalar] = 1
        if not np.array_equal(cardinal_dir_one_hot, self.last_feedback):
            gave_feedback = True
            self.past_given_feedback = self.last_feedback
        else:
            gave_feedback = False
        self.last_feedback = cardinal_dir_one_hot.copy()
        return cardinal_dir_one_hot, gave_feedback

    # TODO: THIS IS NO IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True
