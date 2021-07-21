import numpy as np
from oracle.teacher import Teacher


class WaypointCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(WaypointCorrections, self).__init__(*args, **kwargs)
        self.next_state_coords = self.empty_feedback()

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.zeros(2) - 1

    def give_feedback(self, env):
        """
        Return the expert action from the previous timestep.
        """
        waypoints = env.waypoint_controller.waypoints
        if len(waypoints) == 0:  # already succeeded
            waypoint = np.array(env.get_target())
        else:
            waypoint = waypoints[0].copy()
        gave_feedback = self.past_timestep_feedback is None or not np.array_equal(waypoint, self.past_timestep_feedback)
        self.past_given_feedback = self.last_feedback
        self.last_feedback = waypoint.copy()
        self.gave_feedback = gave_feedback
        # # Divide by 15 so the numbers aren't too big.
        return waypoint / 15, gave_feedback