import numpy as np
from babyai.oracle.teacher import Teacher


class SubgoalSimpleCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(SubgoalSimpleCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return -1 * np.ones(18)

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=18)

    def compute_feedback(self, _, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        sub = np.array(self.next_subgoal)
        return np.concatenate([sub[:7], sub[19:]])

    def success_check(self, state, action, oracle):
        """ Assume the agent completed the subgoal when the last subgoal is no longer in the stack. """
        stack = oracle.stack
        for subgoal in stack:
            sub = oracle.subgoal_to_index(subgoal)
            if np.allclose(self.last_feedback, np.concatenate([sub[:7], sub[19:]])):
                return False
        return True

    def give_feedback(self, state, action, oracle):
        """
        Augment the agent's state observation with teacher feedback.
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        feedback = self.compute_feedback(None)
        gave_feedback = self.last_feedback is None or not np.array_equal(feedback[:19], self.last_feedback[:19])
        self.last_feedback = feedback
        return feedback, gave_feedback

    def step(self, agent_action, oracle):
        """
        Steps the oracle's internal state forward with the agent's current action.
        :param agent_action: The action the agent plans to take.
        """
        self.last_action = self.next_action
        oracle, replan_output = self.replan(oracle, agent_action)
        self.next_action, next_subgoal = replan_output
        # Don't include open subgoals
        if not np.argmax(next_subgoal[:7]) == 1:
            self.next_subgoal = next_subgoal
        self.last_step_error = False
        self.steps_since_lastfeedback += 1
        return oracle