import random

import numpy as np
from oracle.teacher import Teacher


class OffsetWaypointCorrections(Teacher):
    def __init__(self, noise_level=0, noise_duration=1, *args, **kwargs):
        self.noise_level = noise_level
        self.noise_duration = noise_duration
        self.noise_count = 0
        self.waypoint_offset = np.array([0, 0])
        super().__init__(*args, **kwargs)
        self.next_state_coords = self.empty_feedback()

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
        print("Giving feedback", self.noise_level, self.noise_duration, self.waypoint_offset)
        if self.noise_count >= self.noise_duration:
            # Decide if we want noise
            if np.random.uniform() < self.noise_level:
                self.waypoint_offset = random.choice([np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])])
            else:
                self.waypoint_offset = np.array([0, 0])
        self.noise_count += 1
        waypoints = env.waypoint_controller.waypoints
        if len(waypoints) == 0:  # already succeeded
            waypoint = np.array(env.get_target())
        else:
            waypoint = waypoints[0].copy()
        gave_feedback = self.past_timestep_feedback is None or not np.array_equal(waypoint, self.past_timestep_feedback)
        self.past_given_feedback = self.last_feedback
        self.last_feedback = waypoint.copy()
        self.gave_feedback = gave_feedback
        return waypoint - env.get_pos() + self.waypoint_offset, gave_feedback

    # TODO: THIS IS NO IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True

    def reset(self, env):
        super().reset(env)
