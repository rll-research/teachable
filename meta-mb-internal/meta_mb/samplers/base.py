from meta_mb.utils import utils
from meta_mb.logger import logger
from meta_mb.utils.serializable import Serializable
import time
import numpy as np
import copy
from pyprind import ProgBar
from scipy.stats import entropy


class BaseSampler(object):
    """
    Sampler interface

    Args:
        env (gym.Env) : environment object
        policy (meta_mb.policies.policy) : policy object
        batch_size (int) : number of trajectories per task
        max_path_length (int) : max number of steps per trajectory
    """

    def __init__(self, env, policy, num_rollouts, max_path_length):
        assert hasattr(env, 'reset') and hasattr(env, 'step')

        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length

        self.total_samples = num_rollouts * max_path_length
        self.total_timesteps_sampled = 0

    def obtain_samples(self, log=False, log_prefix='', random=False):
        raise NotImplementedError


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])


class SampleProcessor(object):
    """
    Sample processor interface
        - fits a reward baseline (use zero baseline to skip this step)
        - performs Generalized Advantage Estimation to provide advantages (see Schulman et al. 2015 - https://arxiv.org/abs/1506.02438)

    Args:
        baseline (Baseline) : a reward baseline object
        discount (float) : reward discount factor
        gae_lambda (float) : Generalized Advantage Estimation lambda
        normalize_adv (bool) : indicates whether to normalize the estimated advantages (zero mean and unit std)
        positive_adv (bool) : indicates whether to shift the (normalized) advantages so that they are all positive
    """

    def __init__(
            self,
            discount=0.99,
            gae_lambda=1,
            normalize_adv=False,
            positive_adv=False,
    ):

        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv

    def process_samples(self, paths, log=False, log_prefix='', log_teacher=True):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths (list): A list of paths of size (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (dict) : Processed sample data of size [7] x (batch_size x max_path_length)
        """
        assert type(paths) == list, 'paths must be a list'
        assert paths[0].keys() >= {'observations', 'actions', 'rewards'}
        assert self.baseline, 'baseline must be specified - use self.build_sample_processor(baseline_obj)'

        # fits baseline, compute advantages and stack path data
        original_paths = paths
        samples_data, paths = self._compute_samples_data(paths)

        # 7) log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix=log_prefix, log_teacher=log_teacher, original_paths=original_paths)

    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)
        for idx, path in enumerate(paths):
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)

        # # 2) fit baseline estimator using the path returns and predict the return baselines
        # self.baseline.fit(paths, target_key="returns")
        # all_path_baselines = [self.baseline.predict(path) for path in paths]
        #
        # # 3) compute advantages and adjusted rewards
        all_path_baselines = None
        paths = self._compute_advantages(paths, all_path_baselines)

        # 4) stack path data
        observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos = self._concatenate_path_data(
            paths)

        # 5) if desired normalize / shift advantages
        # if self.normalize_adv:
        #     advantages = utils.normalize_advantages(advantages)
        # if self.positive_adv:
        #     advantages = utils.shift_advantages_to_positive(advantages)

        # 6) create samples_data object
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )

        return samples_data, paths

    def _log_path_stats(self, paths, log=False, log_prefix='', log_teacher=True):

        average_discounted_return = np.mean([path["returns"][0] for path in paths])
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        path_length = [path['env_infos']['episode_length'][-1] for path in paths]
        success = [path['env_infos']['success'][-1] for path in paths]
        total_success = [np.sum(path['env_infos']['success']) for path in paths]
        action_entropy = [entropy(step) for path in paths for step in path['agent_infos']['probs']]

        if log_teacher:
            actions_taken = np.array([step for path in paths for step in path['actions']])
            actions_teacher = np.array([step[0] for path in paths for step in path['env_infos']['teacher_action']])
            probs = [probs for path in paths for probs in path['agent_infos']['probs']]
            prob_actions_teacher = [p[int(i)] for p, i in zip(probs, actions_teacher)]
            prob_actions_taken = [p[int(i)] for p, i in zip(probs, actions_taken)]
            logger.logkv(log_prefix + 'ProbActionTeacher', np.mean(prob_actions_teacher))
            logger.logkv(log_prefix + 'ProbActionTaken', np.mean(prob_actions_taken))

            teacher_suggestions = actions_taken == actions_teacher

            # Split by token
            unique_tokens = np.unique([timestep for path in paths for timestep in path['env_infos']['teacher_action']])
            for token in unique_tokens:
                actions_taken = np.array([step for path in paths for step in path['actions']])
                actions_teacher = np.array([step[0] for path in paths for step in path['env_infos']['teacher_action']])
                indices = actions_teacher == token
                teacher_suggestions = actions_taken[indices] == actions_teacher[indices]
                mean_advice = np.mean(teacher_suggestions)
                logger.logkv(log_prefix + 'Advice' + str(token), np.mean(mean_advice))

        if log == 'reward':
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))

        elif log == 'all' or log is True:

            all_actions = np.array([step for path in paths for step in path['actions']])
            logger.logkv(log_prefix + 'ActionDiversity', np.mean(all_actions))

            logger.logkv(log_prefix + 'AverageDiscountedReturn', average_discounted_return)
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.logkv(log_prefix + 'NumTrajs', len(paths))
            logger.logkv(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.logkv(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.logkv(log_prefix + 'MinReturn', np.min(undiscounted_returns))

            logger.logkv(log_prefix + 'AverageSuccess', np.mean(success))
            logger.logkv(log_prefix + 'TotalSuccess', np.mean(total_success))

            logger.logkv(log_prefix + 'AveragePathLength', np.mean(path_length))
            logger.logkv(log_prefix + 'MinPathLength', np.min(path_length))
            logger.logkv(log_prefix + 'MaxPathLength', np.max(path_length))
            logger.logkv(log_prefix + 'StdPathLength', np.std(path_length))

            logger.logkv(log_prefix + 'AvgEntropy', np.mean(action_entropy))
            if log_teacher:
                logger.logkv(log_prefix + 'AvgTeacherAdviceTaken', np.mean(teacher_suggestions))

        return np.mean(undiscounted_returns)

    def _compute_advantages(self, paths, all_path_baselines):
        # assert len(paths) == len(all_path_baselines)

        for idx, path in enumerate(paths):
            path['advantages'] = path['actions']  # TODO: REMOVE THIS! IT'S JUST A PLACEHOLDER SINCE THE REAL ADVANTAGES GET RECOMPUTED LATER


            # path_baselines = np.append(all_path_baselines[idx], 0)
            # deltas = path["rewards"] + \
            #          self.discount * path_baselines[1:] - \
            #          path_baselines[:-1]
            # path["advantages"] = utils.discount_cumsum(
            #     deltas, self.discount * self.gae_lambda)

        return paths

    def _concatenate_path_data(self, paths):
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        dones = np.concatenate([path["dones"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        env_infos = utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        return observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos

    def _stack_path_data(self, paths):
        max_path = max([len(path['observations']) for path in paths])

        observations = self._stack_padding(paths, 'observations', max_path)
        actions = self._stack_padding(paths, 'actions', max_path)
        rewards = self._stack_padding(paths, 'rewards', max_path)
        dones = self._stack_padding(paths, 'dones', max_path)
        returns = self._stack_padding(paths, 'returns', max_path)
        advantages = self._stack_padding(paths, 'advantages', max_path)
        env_infos = utils.stack_tensor_dict_list([path["env_infos"] for path in paths], max_path)
        agent_infos = utils.stack_tensor_dict_list([path["agent_infos"] for path in paths], max_path)

        return observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos

    def _stack_padding(self, paths, key, max_path):
        padded_array = np.stack([
            np.concatenate([path[key], np.zeros((max_path - path[key].shape[0],) + path[key].shape[1:])])
            for path in paths
        ])
        return padded_array
