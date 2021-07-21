import numpy as np
from envs.babyai.oracle.teacher import Teacher
import copy

class SubgoalCorrections(Teacher):
    def __init__(self, *args, **kwargs):
        super(SubgoalCorrections, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return -1 * np.ones(21)

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=21)

    def compute_feedback(self, _, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        return np.array(self.next_subgoal)

    def success_check(self, state, action, oracle):
        """ Assume the agent completed the subgoal when the last subgoal is no longer in the stack. """
        # If we took the wrong last action, we're probably wrong
        if not action == self.next_action:
            return False
        stack = oracle.stack
        for subgoal in stack:
            if np.allclose(self.last_feedback, oracle.subgoal_to_index(subgoal)):
                return False
        return True

    def give_feedback(self, state, action, oracle):
        """
        Augment the agent's state observation with teacher feedback.
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        feedback = self.compute_feedback(None)
        env = oracle.mission
        gave_feedback = self.last_feedback is None or not np.array_equal(feedback, self.last_feedback)
        self.last_feedback = feedback
        subgoal = copy.deepcopy(feedback)
        # Add offset
        if np.array_equal(subgoal[-2:], np.array([-1, -1])):
            print("weird subgoal")
        else:
            subgoal[-2:] = (subgoal[-2:] - env.agent_pos) / 10
        subgoal = np.concatenate([subgoal, (env.agent_pos - 12) / 12, [env.agent_dir / 3]])
        return subgoal, gave_feedback
