import numpy as np
from babyai.oracle.teacher import Teacher

class DemoCorrections(Teacher):
    def reset(self):
        self.env.compute_obj_infos()
        empty_path = np.zeros((self.env.grid.height + self.env.grid.width, 2))
        path = self.oracle.shortest_path_obj()
        empty_path[:len(path)] = path
        self.init_obj_infos = self.env.obj_infos.copy()
        self.demo_path = empty_path.reshape(-1).copy()

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        # Size - obj infos, demos
        return np.zeros_like(np.concatenate([self.init_obj_infos, self.demo_path]))

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        raise NotImplementedError('random feedback not implemented')

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        return np.concatenate([self.init_obj_infos, self.demo_path])

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.

        return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])