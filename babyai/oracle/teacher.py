import numpy as np
import pickle
import copy
import torch

class Teacher:
    """
    Oracle which gives feedback.  Mostly a wrapper around the BabyAI bot class.
    """
    def __init__(self, botclass, env, device=None, feedback_type='oracle', feedback_always=False, cartesian_steps=5, feedback_frequency=1):
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
        oracle = botclass(env)
        self.action_space = env.action_space
        self.botclass = botclass
        self.last_action = -1
        self.next_action, self.next_subgoal = oracle.replan(-1)
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
        self.coords_after_stepping = None
        self.gave_feedback = False
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

    def set_feedback_type(self, feedback_type):
        """
        Specify what feedback type to give.  Currently supported options:
        'oracle' - oracle feedback
        'random' - random actions
        'none' - empty action
        """
        self.feedback_type = feedback_type

    def step(self, agent_action, oracle):
        """
        Steps the oracle's internal state forward with the agent's current action.
        :param agent_action: The action the agent plans to take.
        """
        vis_mask = oracle.vis_mask
        new_oracle = self.botclass(oracle.mission)
        drop_off = len(oracle.stack) > 0 and oracle.mission.carrying and oracle.stack[-1].reason == 'DropOff' and \
                   (not agent_action == oracle.mission.actions.toggle)
        self.last_action = self.next_action
        if drop_off:
            self.next_action, self.next_subgoal = oracle.replan(agent_action)
        else:
            new_oracle.vis_mask = vis_mask
            new_oracle.step = oracle.step
            self.next_action, self.next_subgoal = new_oracle.replan(-1)
            oracle = new_oracle
        self.last_step_error = False
        self.steps_since_lastfeedback += 1
        return oracle

    def step_away_state(self, env_copy, oracle, steps):
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
            env_copy.teacher = None
            if next_action == -1:  # TODO: Is this ever triggered?  As far as I can tell, no.
                next_state = env_copy.gen_obs()['obs']
            else:
                next_state,  rew,  done,  info = env_copy.step(next_action)
                next_state = next_state['obs']
        self.coords_after_stepping = (env_copy.agent_pos.copy(), env_copy.agent_dir)
        return next_state

    def give_feedback(self, state, oracle):
        """
        Augment the agent's state observation with teacher feedback.
        :param oracle:
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        if self.feedback_always:
            feedback = self.compute_feedback(oracle)
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
                feedback = self.compute_feedback(oracle)
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

    def compute_feedback(self, _):
        """
        Returns feedback for the agent as a tensor.
        """
        raise NotImplementedError

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
        oracle = self.botclass(oracle.mission)
        self.next_action, self.next_subgoal = oracle.replan()
        self.last_action = -1
        self.steps_since_lastfeedback = 0
        self.last_feedback = self.empty_feedback()
        self.past_timestep_feedback = self.empty_feedback()
        return oracle