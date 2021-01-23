import numpy as np
import pickle as pkl
from babyai.oracle.teacher import Teacher
import copy


class PreActionAdviceMultipleCopy(Teacher):
    def __init__(self, *args, **kwargs):
        self.action_list = [-1, -1, -1]
        super(PreActionAdviceMultipleCopy, self).__init__(*args, **kwargs)
        self.next_state_coords = self.empty_feedback()

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.concatenate([self.one_hotify(-1) for action in self.action_list])
        # return np.concatenate([self.one_hotify(action) for action in self.action_list] + [np.array([self.steps_since_lastfeedback])])
        action = -1 if self.steps_since_lastfeedback in [-1, None] else self.action_list[self.steps_since_lastfeedback]
        return self.one_hotify(action)
        # return np.concatenate([self.one_hotify(action), np.array([self.steps_since_lastfeedback])])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.concatenate([self.one_hotify(self.action_space.sample()) for _ in range(self.cartesian_steps)])
        # return np.concatenate([self.one_hotify(self.action_space.sample()) for _ in range(self.cartesian_steps)] + [np.array([self.steps_since_lastfeedback])])

    def compute_feedback(self, oracle, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        # Copy so we don't mess up the state of the real oracle
        oracle_copy = pkl.loads(pkl.dumps(oracle))
        self.step_ahead(oracle_copy, last_action=last_action)
        return np.concatenate([self.one_hotify(action) for action in self.action_list])
        # return np.concatenate([self.one_hotify(action) for action in self.action_list] + [np.array([self.steps_since_lastfeedback])])
        action = -1 if self.steps_since_lastfeedback in [-1, None] else self.action_list[self.steps_since_lastfeedback]
        return self.one_hotify(action)
        # return np.concatenate([self.one_hotify(action), np.array([self.steps_since_lastfeedback])])

    def one_hotify(self, index):
        correction = np.zeros((self.action_space.n + 1,))
        correction[index] = 1.0
        return correction

    # TODO: THIS IS NO IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True

    def step_ahead(self, oracle, last_action=-1):
        # Remove teacher so we don't end up with a recursion error
        oracle.mission.teacher = None
        try:
            self.action_list = self.step_away_actions(oracle, self.cartesian_steps, last_action=last_action)
        except Exception as e:
            print("STEP AWAY FAILED PA-M!", e)
            self.action_list = [-1] * self.cartesian_steps
        return oracle

    def step_away_actions(self, oracle, steps, last_action=-1):
        env = oracle.mission
        actions = []
        for step in range(steps):
            oracle, replan_output = self.replan(oracle, last_action)
            last_action = replan_output[0]
            actions.append(last_action)
            env.step(last_action)
        return actions
