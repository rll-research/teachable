import numpy as np
from envs.babyai.oracle.teacher import Teacher

class LandmarkCorrection(Teacher):

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([-1, -1])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        raise NotImplementedError('random feedback not implemented')

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        # TODO: Unhardocde this
        # Hardcoded 1 time-step away
        # Iterate through the objects and order them by their distance from the current object
        # Pick the first one that is closer to the goal than the current object. If none, then return the goal
        dist_pos = np.array(self.env.dist_pos)
        # Distance agent to objects
        agentobj_distances = np.sum(np.abs(dist_pos - self.env.agent_pos), axis=1)
        # Distance agent to goal
        curr_dist = np.sum(np.abs(self.env.obj_pos - self.env.agent_pos))
        # Distance object to goal
        goalobj_distances = np.sum(np.abs(dist_pos - self.env.obj_pos), axis=1)
        idx_closer = np.where(goalobj_distances < curr_dist)
        if len(idx_closer[0]) == 0:
            return np.array([self.env.obj_color, self.env.obj_type])
        else:
            idx_agentobj = range(len(agentobj_distances))
            idx_agentobj = [x for _,x in sorted(zip(agentobj_distances, idx_agentobj))]
            for idx in idx_agentobj:
                if idx in idx_closer[0]:
                    break
            return np.array([self.env.dist_colors[idx], self.env.dist_types[idx]])

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.

        return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])



