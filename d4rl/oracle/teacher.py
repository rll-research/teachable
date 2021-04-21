import numpy as np
import copy
import torch
from d4rl_content.pointmaze.waypoint_controller import WaypointController


class Teacher:
    """
    Oracle which gives feedback.  Mostly a wrapper around the BabyAI bot class.
    """

    def __init__(self, env, device=None, cartesian_steps=1, feedback_frequency=1, controller=None, **kwargs):
        self.action_space = env.action_space
        self.waypoint_controller = controller
        self.last_action = -1
        self.next_action, self.waypoints = None, None
        self.steps_since_lastfeedback = 0
        self.cartesian_steps = cartesian_steps
        self.feedback_frequency = feedback_frequency
        self.kwargs = kwargs
        self.last_feedback = None
        self.past_timestep_feedback = None
        self.past_given_feedback = None
        self.device = device
        self.last_step_error = False
        self.gave_feedback = False
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

    def step(self, env):
        """
        Steps the oracle's internal state forward with the agent's current action.
        :param agent_action: The action the agent plans to take.
        """
        self.last_action = self.next_action
        self.next_action, self.waypoints = self.replan(env)
        self.last_step_error = False
        self.steps_since_lastfeedback += 1
        self.past_timestep_feedback = self.last_feedback

    def replan(self, env):
        action, _ = self.waypoint_controller.get_action(env.get_pos(), env.get_vel(), env.get_target())
        return action, self.waypoint_controller.waypoints.copy()

    def give_feedback(self, env):
        if self.feedback_condition():
            feedback = self.compute_feedback(env)
            gave_feedback = True
            self.past_given_feedback = self.last_feedback
            self.last_feedback = feedback
        else:
            feedback = self.empty_feedback()
            gave_feedback = False
        self.gave_feedback = gave_feedback
        return feedback, gave_feedback

    def empty_feedback(self):
        """
        Empty feedback, used by default if no specific feedback is provided. Returned as a tensor
        """
        raise NotImplementedError

    def compute_feedback(self, env):
        """
        Returns feedback for the agent as a tensor.
        """
        raise NotImplementedError

    def get_last_feedback_indicator(self):
        vec = np.zeros(self.feedback_frequency)
        vec[self.steps_since_lastfeedback] = 1
        return vec

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently gives feedback at a fixed interval, but we could use other strategies (e.g. whenever the agent messes up)
        """
        if self.steps_since_lastfeedback % self.feedback_frequency == 0:
            self.steps_since_lastfeedback = 0
            return True
        else:
            return False

    def reset(self, env):
        self.next_action, self.next_subgoal = self.replan(env)
        self.last_action = -1
        self.steps_since_lastfeedback = 0
        self.last_feedback = self.empty_feedback()
        self.past_timestep_feedback = self.empty_feedback()
