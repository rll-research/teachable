import torch
import numpy as np
import pickle
import copy

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
        #  reinitialize the oracle every timestep  Later it would be better to fix the bot, or at least
        #  figure out what situations it fails and not generate those.
        self.oracle = botclass(env)
        self.env = env
        self.botclass = botclass
        self.last_action = -1
        self.next_action, self.next_subgoal = self.oracle.replan(-1)
        # This first one is going to be wrong
        self.next_state = self.env.gen_obs()
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
        self.oracle_actions.append(self.last_action)
        new_oracle = self.botclass(self.env)
        new_oracle.vis_mask = self.oracle.vis_mask
        new_oracle.step = self.oracle.step
        self.oracle = new_oracle
        self.last_action = self.next_action
        self.next_action, self.next_subgoal = self.oracle.replan(-1)
        self.env_copy1 = pickle.loads(pickle.dumps(self.env))
        self.env_copy1.teacher = None
        if self.next_action == -1:
            self.next_state = self.env.gen_obs()
        else:
            self.next_state,  _,  _,  _ = self.env_copy1.step(self.next_action)

    def compute_full_path(self, steps):
        # Settings steps to -1 computes the full path forward
        before_pos = self.env.agent_pos.copy()
        self.env_copy = pickle.loads(pickle.dumps(self.env))

        new_oracle = self.botclass(self.env_copy)
        new_oracle.vis_mask = copy.deepcopy(self.oracle.vis_mask)
        self.oracle = new_oracle

        # Do the full planning
        env_states = []
        env_rewards = []
        agent_positions = []
        done = False
        self.oracle.mission.teacher = None
        steps_taken = 0
        while not done:
            if steps_taken == steps:
                break
            action, _ = self.oracle.replan(-1)
            obs, reward, done, info = self.oracle.mission.step(action)
            env_states.append(obs)
            env_rewards.append(reward)
            agent_positions.append(self.oracle.mission.agent_pos.copy())
            steps_taken += 1
        after_pos = self.env.agent_pos.copy()
        assert np.all(after_pos == before_pos), 'POSITION CHANGED'
        return np.array(env_states), np.array(env_rewards), np.array(agent_positions)


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
        self.next_action, self.next_subgoal = self.oracle.replan()
        self.last_action = -1
        self.agent_actions = []
        self.oracle_actions = []