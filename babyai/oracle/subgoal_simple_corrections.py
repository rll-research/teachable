import numpy as np
from babyai.oracle.teacher import Teacher
from babyai.oracle.subgoal_corrections import SubgoalCorrections

class SubgoalSimpleCorrections(SubgoalCorrections):
    def __init__(self, *args, **kwargs):
        super(SubgoalSimpleCorrections, self).__init__(*args, **kwargs)

    def step(self, agent_action, oracle):
        """
        Steps the oracle's internal state forward with the agent's current action.
        :param agent_action: The action the agent plans to take.
        """
        self.last_action = self.next_action
        oracle, replan_output = self.replan(oracle, agent_action)
        self.next_action, next_subgoal = replan_output
        # Don't include open subgoals
        if not np.argmax(next_subgoal[:4]) == 0:
            self.next_subgoal = next_subgoal
        self.last_step_error = False
        self.steps_since_lastfeedback += 1
        return oracle