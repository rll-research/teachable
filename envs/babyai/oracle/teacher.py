import numpy as np
import pickle
import copy
import torch


class Teacher:
    """
    Oracle which gives feedback.  Mostly a wrapper around the BabyAI bot class.
    """

    def __init__(self, botclass, env, device=None, feedback_type='oracle', feedback_always=False, cartesian_steps=5,
                 feedback_frequency=1, fully_observed=False):
        """
        :param botclass: Oracle class
        :param env: babyai env
        :param device: 'cuda' or 'cpu'
        :param feedback_type: Specify what feedback type to give. Options: ['oracle', 'random', 'none']
        """
        # TODO: this is pretty sketchy.  To stop the bot from failing, we
        #  reinitialize the oracle every timestep  Later it would be better to fix the bot, or at least
        #  figure out what situations it fails and not generate those.

        self.cartesian_steps = cartesian_steps
        oracle = botclass(env, fully_observed=fully_observed)
        self.action_space = env.action_space
        self.botclass = botclass
        self.last_action = -1
        self.next_action, self.next_subgoal = oracle.get_action(-1)
        # This first one is going to be wrong
        self.next_state = env.gen_obs()['obs']
        self.feedback_type = feedback_type
        self.feedback_always = feedback_always
        self.steps_since_lastfeedback = 0
        self.feedback_frequency = feedback_frequency
        self.last_feedback = None
        self.past_timestep_feedback = None
        self.device = device
        self.last_step_error = False
        self.gave_feedback = False
        self.fully_observed = fully_observed
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

    def step(self, agent_action, oracle):
        """
        Steps the oracle's internal state forward with the agent's current action.
        :param agent_action: The action the agent plans to take.
        """
        self.last_action = self.next_action
        oracle, replan_output = self.replan(oracle, agent_action)
        self.next_action, self.next_subgoal = replan_output
        self.last_step_error = False
        self.steps_since_lastfeedback += 1
        return oracle

    def replan(self, oracle, last_action):
        env = oracle.mission
        # Generally we create a fresh oracle each time to prevent the teacher from getting stuck telling the agent
        # trying to undo old actions rather than correcting it from where it starts.
        # However, when we're dropping an object off to unblock a path we need to keep the existing oracle
        # so the agent doesn't lose track of why it's doing this and where it wants to drop it.
        drop_off = len(oracle.stack) > 0 and env.carrying and oracle.stack[-1].reason == 'DropOff' and \
                   (not last_action == env.actions.toggle)
        if drop_off or self.next_action == last_action:
            replan_output = oracle.get_action(last_action)
        else:
            new_oracle = self.botclass(env, rng=copy.deepcopy(oracle.rng), fully_observed=self.fully_observed)
            new_oracle.vis_mask = oracle.vis_mask.copy()
            new_oracle.step = oracle.step
            replan_output = new_oracle.get_action(-1)
            oracle = new_oracle
        return oracle, replan_output

    def step_away_state(self, oracle, steps, last_action=-1):
        env = oracle.mission
        actions = []
        for step in range(steps):
            oracle, replan_output = self.replan(oracle, last_action)
            last_action = replan_output[0]
            actions.append(last_action)
            next_state, rew, done, info = env.step(last_action)
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
        if self.feedback_always:
            feedback = self.compute_feedback(oracle, last_action)
            self.past_timestep_feedback = self.last_feedback
            self.last_feedback = feedback
            gave_feedback = True
        elif self.feedback_type == 'none':
            feedback = self.empty_feedback()
            gave_feedback = False
        elif self.feedback_type == 'random':
            feedback = self.random_feedback()
            gave_feedback = True
        elif self.feedback_type == 'oracle':
            if self.feedback_condition():
                feedback = self.compute_feedback(oracle, last_action)
                gave_feedback = True
                self.past_timestep_feedback = self.last_feedback
                self.last_feedback = feedback
            else:
                feedback = self.empty_feedback()
                gave_feedback = False
        else:
            raise ValueError("Unsupported feedback type")
        self.gave_feedback = gave_feedback
        return feedback, gave_feedback

    def empty_feedback(self):
        """
        Empty feedback, used by default if no specific feedback is provided. Returned as a tensor
        """
        raise NotImplementedError

    def compute_feedback(self, obs, action):
        """
        Returns feedback for the agent as a tensor.
        """
        raise NotImplementedError

    def get_last_feedback_indicator(self):
        vec = np.zeros(self.feedback_frequency)
        vec[self.steps_since_lastfeedback] = 1
        return vec

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently gives feedback at a fixed interval, but we could use other strategies (e.g. whenever the agent messes up)
        """
        if self.steps_since_lastfeedback % self.feedback_frequency == 0:
            self.steps_since_lastfeedback = 0
            return True
        else:
            return False

    def reset(self, oracle):
        oracle = self.botclass(oracle.mission, rng=copy.deepcopy(oracle.rng), fully_observed=self.fully_observed)
        self.next_action, self.next_subgoal = oracle.get_action()
        self.last_action = -1
        self.steps_since_lastfeedback = 0
        self.last_feedback = self.empty_feedback()
        self.past_timestep_feedback = self.empty_feedback()
        return oracle
