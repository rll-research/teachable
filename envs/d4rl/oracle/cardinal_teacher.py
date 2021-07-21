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

    def give_feedback(self, env):
        # Always re-compute feedback, but only count it as new feedback if it changes
        action = self.next_action
        # We're on a corner. In that case, the "action" is replaced with the vector to the next waypoint,
        # or 2 waypoints ahead if we're already close
        if np.abs(self.next_action[0]) == np.abs(self.next_action[1]) == 1:
            action = self.waypoints[0] - env.get_pos()
            if len(self.waypoints) >= 2 and np.linalg.norm(action - env.get_pos()) < .5:
                action = self.waypoints[1] - env.get_pos()
        up_dist = action[1]
        down_dist = -up_dist
        right_dist = action[0]
        left_dist = -right_dist
        cardinal_dir_scalar = np.argmax([left_dist, up_dist, right_dist, down_dist])
        cardinal_dir_one_hot = np.zeros(4)
        cardinal_dir_one_hot[cardinal_dir_scalar] = 1
        if not np.array_equal(cardinal_dir_one_hot, self.last_feedback):
            gave_feedback = True
            self.past_given_feedback = self.last_feedback
        else:
            gave_feedback = False
        self.last_feedback = cardinal_dir_one_hot.copy()
        return cardinal_dir_one_hot, gave_feedback

