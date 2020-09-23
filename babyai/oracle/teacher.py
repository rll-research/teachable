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
        self.env = env
        self.botclass = botclass
        self.last_action = -1
        self.next_action, self.next_subgoal = oracle.replan(-1)
        # This first one is going to be wrong
        self.next_state = env.gen_obs()
        self.agent_actions = []
        self.oracle_actions = []
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
        self.agent_actions.append(agent_action)
        self.oracle_actions.append(self.last_action)
        new_oracle = self.botclass(oracle.mission)
        new_oracle.vis_mask = oracle.vis_mask
        new_oracle.step = oracle.step
        oracle = new_oracle
        self.last_action = self.next_action
        self.last_step_error = False
        try:
            self.next_action, self.next_subgoal = oracle.replan(-1)
        except:
            self.next_action = -1
            print("NOT UPDATING ACTION AND SUBGOAL")
            self.last_step_error = True
        self.env_copy1 = pickle.loads(pickle.dumps(self.env))
        self.env_copy1.teacher = None
        try:
            self.next_state = self.step_away_state(self.env_copy1, oracle, self.cartesian_steps)
        except:
            self.next_state = self.next_state * 0
            self.last_step_error = True
        self.steps_since_lastfeedback += 1
        return oracle

    def step_away_state(self, env_copy, oracle, steps):

        for _ in range(steps):
            new_oracle = self.botclass(env_copy)
            new_oracle.vis_mask = oracle.vis_mask.copy()
            new_oracle.step = oracle.step
            new_oracle._process_obs()
            next_action, _ = new_oracle.replan(-1)
            env_copy.teacher = None
            if next_action == -1:
                next_state = env_copy.gen_obs()
            else:
                next_state,  _,  _,  _ = env_copy.step(next_action)            
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
        elif self.feedback_type == 'none' or not self.feedback_condition():
            feedback = self.empty_feedback()
        elif self.feedback_type == 'random':
            feedback = self.random_feedback()
        elif self.feedback_type == 'oracle':
            feedback = self.compute_feedback()
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
        Check whether we should give feedback.
        """
        raise NotImplementedError

    def reset(self, oracle):
        oracle = self.botclass(oracle.mission)
        self.next_action, self.next_subgoal = oracle.replan()
        self.last_action = -1
        self.agent_actions = []
        self.oracle_actions = []
        self.steps_since_lastfeedback = 0
        self.last_feedback = None
        return oracle