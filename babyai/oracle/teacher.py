import torch
import numpy as np

class Teacher:
    """
    Oracle which gives feedback.  Mostly a wrapper around the BabyAI bot class.
    """
    def __init__(self, botclass, env, device=None, feedback_type='oracle', feedback_always=False):
        """
        :param botclass: Oracle class
        :param env: babyai env
        :param device: 'cuda' or 'cpu'
        :param feedback_type: Specify what feedback type to give. Options: ['oracle', 'random', 'none']
        """
        # TODO: this is pretty sketchy.  To stop the bot from failing, we
        #  (a) reinitialize the oracle every timestep, and
        #  (b) open all doors every timestep.  Later it would be better to fix the bot, or at least
        #  figure out what situations it fails and not generate those.
        self.oracle = botclass(env)
        self.env = env
        # self.env.open_all_doors()
        self.botclass = botclass
        self.next_action = -1
        self.agent_actions = []
        self.oracle_actions = []
        self.feedback_type = feedback_type
        self.feedback_always = feedback_always
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

    def step(self, agent_action):
        """
        Steps the oracle's internal state forward with the agent's current action.
        :param agent_action: The action the agent plans to take.
        """
        self.agent_actions.append(agent_action)
        self.oracle_actions.append(self.next_action)
        new_oracle = self.botclass(self.env)
        new_oracle.vis_mask = self.oracle.vis_mask
        self.oracle = new_oracle
        self.next_action = self.oracle.replan(-1)
        # self.path = self.oracle.shortest_path_obj()


    def give_feedback(self, state):
        """
        Augment the agent's state observation with teacher feedback.
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        if self.feedback_always:
            feedback = self.compute_feedback()
        elif self.feedback_type == 'none' or not self.feedback_condition():
            feedback = self.empty_feedback()
        elif self.feedback_type == 'random':
            feedback = self.random_feedback()
        elif self.feedback_type == 'oracle':

            feedback = self.compute_feedback()
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

    def reset(self):
        self.oracle = self.botclass(self.env)
        self.next_action = -1
        self.agent_actions = []
        self.oracle_actions = []