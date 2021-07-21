import numpy as np
import torch

class Teacher:
    """
    Oracle which gives feedback.  Mostly a wrapper around the BabyAI bot class.
    """

    def __init__(self, **kwargs):
        self.last_action = -1
        self.next_action, self.waypoints = None, None
        self.steps_since_lastfeedback = 0
        self.last_feedback = None
        self.past_timestep_feedback = None
        self.past_given_feedback = None
        self.gave_feedback = False

    def step(self):
        """
        Steps the oracle's internal state forward with the agent's current action.
        :param agent_action: The action the agent plans to take.
        """
        self.steps_since_lastfeedback += 1
        self.past_timestep_feedback = self.last_feedback

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

    def reset(self):
        self.last_feedback = self.empty_feedback()
        self.past_timestep_feedback = self.empty_feedback()
