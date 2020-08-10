import numpy as np
from babyai.oracle.teacher import Teacher
import pickle

class CartesianCorrections(Teacher):

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return -1*np.ones(self.obs_size)

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.random.uniform(0, 1, size=self.obs_size)

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        # TODO: Unhardocde this
        # Hardcoded 1 time-step away
        # self.env_states, self.env_rewards, self.agent_positions = self.compute_full_path(1)
        # if len(self.env_states) > 0:
        #     feedback = self.env_states[0]
        # else:
        #     feedback = -1*np.ones(self.obs_size)
        # return np.array(feedback)
        return np.array(self.next_state)

    def step(self, agent_action):
        super().step(agent_action)
        self.env_copy1 = pickle.loads(pickle.dumps(self.env))
        self.env_copy1.teacher = None
        if self.next_action == -1:
            self.next_state = self.env.gen_obs()
        else:
            self.next_state, _, _, _ = self.env_copy1.step(self.next_action)
        
    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.

        return len(self.agent_actions) > 0 and (not self.agent_actions[-1] == self.oracle_actions[-1])



