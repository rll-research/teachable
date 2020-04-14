import torch

from babyai.oracle.teacher import Teacher



class ActionAdvice(Teacher):

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return torch.FloatTensor([-1]).to(self.device).numpy()

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        return torch.FloatTensor([self.next_action]).numpy()

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.
        return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])



