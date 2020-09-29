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
        self.botclass = botclass
        self.last_action = -1
        self.next_action, self.next_subgoal = oracle.replan(-1)
        # This first one is going to be wrong
        self.next_state = env.gen_obs()
        self.feedback_type = feedback_type
        self.feedback_always = feedback_always
        self.steps_since_lastfeedback = 0
        self.feedback_frequency = feedback_frequency
        self.last_feedback = None
        self.device = device
        self.last_step_error = False
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
        new_oracle = self.botclass(oracle.mission)
        new_oracle.vis_mask = oracle.vis_mask
        new_oracle.step = oracle.step
        oracle = new_oracle
        self.last_action = self.next_action
        self.last_step_error = False
        try:
            self.next_action, self.next_subgoal = oracle.replan(-1)
        except Exception as e:
            self.next_action = -1
            print("NOT UPDATING ACTION AND SUBGOAL")
            print(e)
            print("CURRENT VISMASK", oracle.vis_mask)
            self.last_step_error = True
        oracle.mission.teacher = None
        env_copy1 = pickle.loads(pickle.dumps(oracle.mission))
        env_copy1.teacher = None
        try:
            self.next_state = self.step_away_state(env_copy1, oracle, self.cartesian_steps)
        except Exception as e:
            print("STEP AWAY FAILED!")
            print(e)
            print("CURRENT VISMASK", oracle.vis_mask)
            self.next_state = self.next_state * 0
            self.last_step_error = True
        self.steps_since_lastfeedback += 1
        oracle.mission.teacher = self
        return oracle

    def step_away_state(self, env_copy, oracle, steps):
        for step in range(steps):
            new_oracle = self.botclass(env_copy)
            new_oracle.vis_mask = oracle.vis_mask.copy()
            new_oracle.step = oracle.step
            new_oracle._process_obs()
            next_action, _ = new_oracle.replan(-1)
            oracle = new_oracle
            env_copy.teacher = None
            if next_action == -1:  # TODO: Is this ever triggered?  As far as I can tell, no.
                next_state = env_copy.gen_obs()
            else:
                next_state,  rew,  done,  info = env_copy.step(next_action)
        return next_state

    def give_feedback(self, state):
        """
        Augment the agent's state observation with teacher feedback.
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        if self.feedback_always:
            feedback = self.compute_feedback()
            self.last_feedback = feedback
        elif self.feedback_type == 'none':
            feedback = self.empty_feedback()
        elif self.feedback_type == 'random':
            feedback = self.random_feedback()
        elif self.feedback_type == 'oracle':
            if self.feedback_condition():
                feedback = self.compute_feedback()
            else:
                feedback = self.empty_feedback()
            self.last_feedback = feedback
        else:
            raise ValueError("Unsupported feedback type")
        return feedback

    def empty_feedback(self):
        """
        Empty feedback, used by default if no specific feedback is provided. Returned as a tensor
        """
        raise NotImplementedError

    def compute_feedback(self):
        """
        Returns feedback for the agent as a tensor.
        """
        raise NotImplementedError

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently gives feedback at a fixed interval, but we could use other strategies (e.g. whenever the agent messes up)
        """
        # TODO NOW Fix this
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
        self.last_feedback = None
        return oracle