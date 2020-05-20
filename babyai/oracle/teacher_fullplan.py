import torch
import numpy as np
import copy

class TeacherFullPlan:
    """
    Oracle which gives feedback.  Mostly a wrapper around the BabyAI bot class.
    """
    def __init__(self, botclass, env, env_copy=None, device=None, feedback_type='oracle'):
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
        self.env_copy = env_copy
        self.env.open_all_doors()
        self.botclass = botclass
        self.next_action = None
        self.agent_actions = []
        self.oracle_actions = []
        self.feedback_type = feedback_type
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
        # TODO: Currently we open all doors to prevent planning errors. We should fix this.
        self.env.open_all_doors()
        self.agent_actions.append(agent_action)
        self.oracle_actions.append(self.next_action)
        new_oracle = self.botclass(self.env)
        new_oracle.vis_mask = self.oracle.vis_mask
        self.oracle = new_oracle
        self.next_action = self.oracle.replan()
        self.path = self.oracle.shortest_path_obj()
        self.env_path, self.env_rewards, self.agent_positions = self.compute_full_path()
        import IPython
        IPython.embed()
        
    def compute_full_path(self):
        self.env_copy = copy.deepcopy(self.env)
        new_oracle = self.botclass(self.env_copy)
        new_oracle.vis_mask = copy.deepcopy(self.oracle.vis_mask)
        self.oracle = new_oracle

        # Do the full planning
        env_states = []
        env_rewards = []
        agent_positions = []
        done = False
        self.oracle.mission.teacher = None
        while not done:
            action = self.oracle.replan()
            obs, reward, done, info = self.oracle.mission.step(action)
            env_states.append(obs)
            env_rewards.append(reward)
            agent_positions.append(self.oracle.mission.agent_pos.copy())

        return np.array(env_states), np.array(env_rewards), np.array(agent_positions)


    def give_feedback(self, state):
        """
        Augment the agent's state observation with teacher feedback.
        :param state: Agent's current observation as a dictionary
        :return: Same dictionary with feedback in the "feedback" key of the dictionary
        """
        if self.feedback_type == 'none' or not self.feedback_condition():
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
        pass