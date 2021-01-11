import numpy as np
import pickle as pkl
from babyai.oracle.teacher import Teacher
import copy


class PreActionAdviceMultiple(Teacher):
    def __init__(self, *args, **kwargs):
        super(PreActionAdviceMultiple, self).__init__(*args, **kwargs)
        self.next_state_coords = self.empty_feedback()

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

    # TODO: THIS IS NO IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True

    def step_ahead(self, oracle):
        original_teacher = oracle.mission.teacher
        oracle.mission.teacher = None
        env = pkl.loads(pkl.dumps(oracle.mission))
        env.teacher = None
        try:
            self.action_list = self.step_away_actions(env, oracle, self.cartesian_steps)
        except Exception as e:
            print("STEP AWAY FAILED!", e)
            self.action_list = [-1] * self.cartesian_steps
        oracle.mission.teacher = original_teacher
        return oracle

    def step_away_actions(self, env_copy, oracle, steps):
        next_action = -1
        actions = []
        for step in range(steps):
            vis_mask = oracle.vis_mask.copy()
            new_oracle = self.botclass(env_copy)
            drop_off = len(oracle.stack) > 0 and oracle.mission.carrying and oracle.stack[-1].reason == 'DropOff' and \
                       (not next_action == oracle.mission.actions.toggle)
            if drop_off:
                next_action, next_subgoal = copy.deepcopy(oracle).replan(next_action)
            else:
                new_oracle.vis_mask = vis_mask
                new_oracle.step = oracle.step
                new_oracle._process_obs()
                next_action, next_subgoal = new_oracle.replan(-1)
                oracle = new_oracle
            env_copy.teacher = None
            actions.append(next_action)
            if next_action == -1:
                assert False, "It was triggered after all"
            else:
                env_copy.step(next_action)
        return actions
