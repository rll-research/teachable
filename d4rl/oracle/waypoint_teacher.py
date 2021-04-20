import numpy as np
from oracle.teacher import Teacher


class WaypointCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(WaypointCorrections, self).__init__(*args, **kwargs)
        self.next_state_coords = self.empty_feedback()
        self.static_waypoints = None

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.zeros(2) - 1

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=2)

    def give_feedback(self, env):
        """
        Return the expert action from the previous timestep.
        """
        agent_pos = env.get_pos()
        if len(self.static_waypoints) == 0:  # already succeeded
            waypoint = np.array(env.get_target())
            gave_feedback = False
        else:
            dist_to_curr_waypoint = np.linalg.norm(agent_pos - self.static_waypoints[0])
            if len(self.waypoints) > 1:
                dist_to_future_waypoint = np.linalg.norm(agent_pos - self.static_waypoints[1])
            else:
                dist_to_future_waypoint = float('inf')
            close_to_curr_waypoint = dist_to_curr_waypoint < .25
            closer_to_next_waypoint = dist_to_future_waypoint < dist_to_curr_waypoint
            if close_to_curr_waypoint or closer_to_next_waypoint:
                self.static_waypoints = self.static_waypoints[1:]
                gave_feedback = True
            else:
                gave_feedback = False
            waypoint = self.static_waypoints[0].copy()
        return waypoint, gave_feedback

    # TODO: THIS IS NO IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True

    def reset(self, env):
        super().reset(env)
        self.static_waypoints = self.waypoints
