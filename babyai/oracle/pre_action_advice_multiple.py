import numpy as np
import pickle as pkl
from babyai.oracle.teacher import Teacher


class PreActionAdviceMultiple(Teacher):
    def __init__(self, *args, **kwargs):
        super(PreActionAdviceMultiple, self).__init__(*args, **kwargs)

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.concatenate([self.one_hotify(-1) for _ in range(self.cartesian_steps)])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.concatenate([self.one_hotify(self.action_space.sample()) for _ in range(self.cartesian_steps)])

    def compute_feedback(self, oracle):
        """
        Return the expert action from the previous timestep.
        """
        self.step_ahead(oracle)
        return np.concatenate([self.one_hotify(action) for action in self.action_list])

    def one_hotify(self, index):
        correction = np.zeros((self.action_space.n + 1,))
        correction[index] = 1.0
        return correction

    def success_check(self, state, action, oracle):
        return True # TODO: do this!
        # opt_action = int(self.last_action)
        # followed_opt_action = (opt_action == action)
        # return followed_opt_action and self.gave_feedback

    def step_ahead(self, oracle):
        original_teacher = oracle.mission.teacher
        oracle.mission.teacher = None
        env_copy1 = pkl.loads(pkl.dumps(oracle.mission))
        env_copy1.teacher = None
        self.action_list = self.step_away_actions(env_copy1, oracle, self.cartesian_steps)
        oracle.mission.teacher = original_teacher
        return oracle

    def step_away_actions(self, env_copy, oracle, steps):
        actions = []
        next_action = -1
        for step in range(steps):
            vis_mask = oracle.vis_mask
            new_oracle = self.botclass(oracle.mission)
            drop_off = len(oracle.stack) > 0 and oracle.mission.carrying and oracle.stack[-1].reason == 'DropOff' and \
                       (not next_action == oracle.mission.actions.toggle)
            if drop_off:
                next_action, next_subgoal = oracle.replan(next_action)
            else:
                new_oracle.vis_mask = vis_mask
                new_oracle.step = oracle.step
                next_action, next_subgoal = new_oracle.replan(-1)
                oracle = new_oracle
            actions.append(next_action)
            env_copy.teacher = None
        return actions
