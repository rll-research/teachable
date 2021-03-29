import numpy as np
import pickle as pkl
from babyai.oracle.teacher import Teacher


class OSREasy(Teacher):
    def __init__(self, *args, **kwargs):
        self.num_steps = 1
        self.next_state_coords = np.zeros(3)
        self.goal_coords = np.array([-1, -1])
        super(OSREasy, self).__init__(*args, **kwargs)

    def empty_feedback(self, env=None):
        """
        Return a tensor corresponding to no feedback.
        """
        return self.generic_feedback(env)

    def generic_feedback(self, env, offset=True):
        if env is None:
            pos = np.array([-1, -1, -1])
        else:
            pos = np.concatenate([(env.agent_pos - 12) / 12, [env.agent_dir / 3]])
        if offset and env:
            coord_offset = self.next_state_coords.copy()
            coord_offset[1:] = coord_offset[1:] - env.agent_pos
        else:
            coord_offset = -np.ones(3)
        return np.concatenate([coord_offset, pos,
                               self.get_last_feedback_indicator()])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.concatenate([np.random.uniform(0, 1, size=3), self.get_last_feedback_indicator()])

    def compute_feedback(self, oracle, last_action=-1):
        """
        Return the expert action from the previous timestep.
        """
        # Copy so we don't mess up the state of the real oracle
        oracle_copy = pkl.loads(pkl.dumps(oracle))
        self.step_ahead(oracle_copy, last_action=last_action)
        env = oracle.mission
        return self.generic_feedback(env)

    # TODO: THIS IS NO IMPLEMENTED FOR THIS TEACHER! IF WE END UP USING THIS METRIC, WE SHOULD MAKE IT CORRECT!
    def success_check(self, state, action, oracle):
        return True

    def step_ahead(self, oracle, last_action=-1):
        env = oracle.mission
        # Remove teacher so we don't end up with a recursion error
        env.teacher = None
        # try:
        num_steps = np.random.randint(2, self.cartesian_steps + 1)
        self.num_steps = num_steps

        self.next_state, next_coords, actions, env = self.step_away_state(oracle, num_steps,
                                                                          last_action=last_action)
        self.goal_coords = next_coords[:2].copy()
        if actions[-1] in [env.actions.drop, env.actions.pickup] or env.done:
            first = 1
            # Position where we'll place the item
            self.goal_coords = self.goal_coords + env.dir_vec
        else:
            first = 0
        self.next_state_coords = np.concatenate([[first], self.goal_coords]).astype(np.float32)
        # except Exception as e:
        #     print("STEP AWAY FAILED Offset!", e)
        #     self.next_state = self.next_state * 0
        #     self.next_state_coords = self.empty_feedback()
        #     self.last_step_error = True
        return oracle

    def step_away_state(self, oracle, steps, last_action=-1):
        env = oracle.mission
        actions = []
        for step in range(steps):
            oracle, replan_output = self.replan(oracle, last_action)
            last_action = replan_output[0]
            actions.append(last_action)
            next_state, rew, done, info = env.step(last_action)
            # End early if we're picking something up or putting it down
            if last_action in [env.actions.drop, env.actions.pickup]:
                break
        next_state = next_state['obs']
        coords = np.concatenate([env.agent_pos, [env.agent_dir, int(env.carrying is not None)]])
        return next_state, coords, actions, env

    def give_feedback(self, state, last_action, oracle):
        """
        Augment the agent's state observation with teacher feedback.
        :param oracle:
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        oracle = pkl.loads(pkl.dumps(oracle))
        env = oracle.mission
        if self.feedback_condition(env, last_action):
            feedback = self.compute_feedback(oracle, last_action)
            gave_feedback = True
            self.past_timestep_feedback = self.last_feedback
            self.last_feedback = feedback
        else:
            feedback = self.empty_feedback(env)
            gave_feedback = False
        self.gave_feedback = gave_feedback
        return feedback, gave_feedback

    def feedback_condition(self, env, action=None):
        """
        Returns true when we should give feedback.
        Currently gives feedback at a fixed interval, but we could use other strategies (e.g. whenever the agent messes up)
        """
        if (self.steps_since_lastfeedback % self.num_steps == 0) or np.array_equal(env.agent_pos, self.goal_coords):
            self.steps_since_lastfeedback = 0
            return True
        return False

    def get_last_feedback_indicator(self):
        try:
            vec = np.zeros(self.feedback_frequency)
            vec[self.steps_since_lastfeedback] = 1
        except Exception as e:
            print("uh oh, looks like we can't choose a feedback freq (OSREasy)", e)
        return vec