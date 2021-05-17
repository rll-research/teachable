from meta_mb.utils.serializable import Serializable
from babyai.levels.iclr19_levels import *
from envs.d4rl_envs import PointMassEnv, AntEnv, PointMassEnvSimple, PointMassEnvSimpleDiscrete
import copy

class Curriculum(Serializable):
    def __init__(self, advance_curriculum_func, env, start_index=0, curriculum_type=0, reward_type='dense', **kwargs):
        """

        :param advance_curriculum_func: Either 'one_hot' or 'smooth' depending on whether you want each level of the
        curriculum to be a single environment or a distribution over past environments
        :param start_index: what index of the curriculum to start on
        :param kwargs: arguments for the environment
        """
        Serializable.quick_init(self, locals())
        reward_env_name = 'sparse' if reward_type == 'sparse' else 'dense'
        self.advance_curriculum_func = advance_curriculum_func
        self.env = env
        self.kwargs = kwargs
        self.reward_type = reward_type
        self.reward_env_name = reward_env_name
        if env == 'point_mass':
            self.levels_list = {k: -1 for k in range(13)}
        elif env == 'ant':
            self.levels_list = {k: -1 for k in range(12)}
        elif env == 'babyai':
            # List of all the levels.  There are actually a bunch more: some ones which were omitted since they were
            # very similar to the current ones (e.g. more Level_GoToLocal variants with different sizes and num dists)
            # also some harder levels with multiple instructions chained together.
            # TODO: consider re-introducing the harder levels, especially as held-out levels
            self.train_levels = [
                # Easy levels; intro all PreAction tokens and most subgoals
                Level_GoToRedBallNoDists(**kwargs),
                # 0 --> intro L, R, Forward PreAction, Explore and GoNextTo subgoals
                Level_GoToRedBallGrey(**kwargs),  # 1 --> first level with distractors
                Level_GoToRedBall(**kwargs),  # 2 --> first level with colored distractors
                Level_GoToObjS5(**kwargs),  # 3 --> first level where the goal is something other than a red ball
                Level_GoToLocalS5N2(**kwargs),  # 4 --> first level where the task means something
                Level_PickupLocalS5N2(**kwargs),  # 5 --> intro Pickup subgoal and pickup PreAction
                Level_PutNextLocalS5N2(**kwargs),  # 6 --> intro Drop subgoal and drop PreAction
                Level_OpenLocalS5N2(**kwargs),  # 7 --> intro Open subgoal and open PreAction

                # Medium levels (here we introduce the harder teacher; no new tasks, just larger sizes)
                Level_GoToObjS7(**kwargs),  # 8
                Level_GoToLocalS7N4(**kwargs),  # 9
                Level_PickupLocalS7N4(**kwargs),  # 10
                Level_PutNextLocalS7N4(**kwargs),  # 11
                Level_OpenLocalS7N4(**kwargs),  # 12

                # Hard levels (bigger sizes, some new tasks)
                Level_GoToObj(**kwargs),  # 13
                Level_GoToLocal(**kwargs),  # 14
                Level_PickupLocal(**kwargs),  # 15
                Level_PutNextLocal(**kwargs),  # 16
                Level_OpenLocal(**kwargs),  # 17

                # Biggest levels (larger grid)
                Level_GoToObjMazeOpen(**kwargs),  # 18
                Level_GoToOpen(**kwargs),  # 19
                Level_GoToObjMazeS4R2(**kwargs),  # 20
                Level_GoToObjMazeS5(**kwargs),  # 21
                Level_Open(**kwargs),  # 22
                Level_GoTo(**kwargs),  # 23
                Level_Pickup(**kwargs),  # 24
                Level_PutNext(**kwargs),  # 25
            ]

            self.held_out_levels = [
                # Larger sizes than we've seen before
                Level_PickupObjBigger(**kwargs),  # 26 test0

                # More distractors than we've seen before
                Level_GoToObjDistractors(**kwargs),  # 27 test1

                # New object
                Level_GoToHeldout(**kwargs),  # 28 test2

                # Task we've seen before, but new instructions
                Level_GoToGreenBox(**kwargs),  # 29 test3
                Level_PutNextSameColor(**kwargs),  # 30 test4

                # New object
                Level_Unlock(**kwargs),  # 31 test5 ("unlock" is a completely new instruction)
                Level_GoToImpUnlock(**kwargs),  # 32 test6
                Level_UnblockPickup(**kwargs),  # 33 test7 (known task, but now there's the extra step of unblocking)
                Level_Seek(**kwargs),  # 34 test8

                # Chain multiple instructions together
                # Level_OpenDoorsDouble(**kwargs),   # TODO: teacher fails
                # Level_GoToDouble(**kwargs),  # TODO: teacher fails

                # Easier heldout levels
                Level_GoToGreenBoxLocal(**kwargs),  # 35 test9
                Level_PutNextSameColorLocal(**kwargs),  # 36 test10
                Level_UnlockLocal(**kwargs),  # 37 test11 ("unlock" is a completely new instruction)
                Level_GoToImpUnlockLocal(**kwargs),  # 38 test12
                Level_SeekLocal(**kwargs),  # 39 test13

                Level_GoToObjDistractorsLocal(**kwargs),  # 40 test14
                Level_GoToSmall2by2(**kwargs),  # 41 test15
                Level_GoToSmall3by3(**kwargs),  # 42 test16
                Level_SeekSmall2by2(**kwargs),  # 43 test17
                Level_SeekSmall3by3(**kwargs),  # 44 test18
                Level_GoToObjDistractorsLocalBig(**kwargs),  # 45 test19
                Level_OpenSmall2by2(**kwargs),  # 46 test20
                Level_OpenSmall3by3(**kwargs),  # 47 test21
                Level_SeekL0(**kwargs),  # 48 test22
            ]
            self.levels_list = self.train_levels + self.held_out_levels

        # If start index isn't specified, start from the beginning (if we're using the pre-levels), or start
        # from the end of the pre-levels.
        if self.advance_curriculum_func == 'four_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[16, 22, 23, 24]] = .25
        elif self.advance_curriculum_func == 'four_big_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[22, 23, 24, 25]] = .25
        elif self.advance_curriculum_func == 'five_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[16, 22, 23, 24, 25]] = .2
        elif self.advance_curriculum_func == 'goto_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[9, 14, 23]] = 1 / 3.
        elif self.advance_curriculum_func == 'easy_goto':
            self.distribution = np.zeros((len(self.levels_list)))
            indices = [4, 9, 14, 7, 12, 17, 39, 40, 41, 42, 43, 44, 46, 47]
            self.distribution[indices] = 1. / len(indices)
        elif advance_curriculum_func == 'uniform':
            prob_mass = 1 / (start_index + 1)
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[:start_index + 1] = prob_mass
        else:
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[start_index] = 1
        # class_name, class_args = self.levels_list[start_index]  # TODO: double check this doesn't do horrible things with the babyai levels
        self.set_wrapped_env(start_index)
        self.index = start_index

    def set_wrapped_env(self, index):
        if not self.levels_list[index] == -1:
            return self.levels_list[index]
        kwargs = self.kwargs
        reward_env_name = self.reward_env_name
        reward_type = self.reward_type
        seed = self.levels_list[index]
        if self.env == 'point_mass':
            if index == 0:
                level = PointMassEnv(f'maze2d-open-{reward_env_name}-v0', reward_type=reward_type, **kwargs)  # 0
            elif index == 1:
                level = PointMassEnv(f'maze2d-umaze-{reward_env_name}-v1', reward_type=reward_type, **kwargs)  # 1
            elif index == 2:
                level = PointMassEnv(f'maze2d-medium-{reward_env_name}-v1', reward_type=reward_type, **kwargs)  # 2
            elif index == 3:
                level = PointMassEnv(f'maze2d-large-{reward_env_name}-v1', reward_type=reward_type, **kwargs)  # 3
            elif index == 4:
                level = PointMassEnv('maze2d-randommaze-v0', reward_type=reward_type, **kwargs)  # 4
            elif index == 5:
                level = PointMassEnv(f'maze2d-umaze-{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 5
                             **kwargs)
            elif index == 6:
                level = PointMassEnv(f'maze2d-medium-{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 6
                             **kwargs)
            elif index == 7:
                level = PointMassEnv(f'maze2d-large-{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 7
                             **kwargs)
            elif index == 8:
                level = PointMassEnv(f'maze2d-umaze-{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 8
                             reset_start=False, **kwargs)
            elif index == 9:
                level = PointMassEnv(f'maze2d-medium-{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 9
                             reset_start=False, **kwargs)
            elif index == 10:
                level = PointMassEnv(f'maze2d-large-{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 10
                             reset_start=False, **kwargs)
            elif index == 11:
                level = PointMassEnv('maze2d-randommaze-7x7-v0', reward_type=reward_type, **kwargs)  # 4
            elif index == 12:
                level = PointMassEnv('maze2d-randommaze-8x8-v0', reward_type=reward_type, **kwargs)  # 4
            else:
                raise NotImplementedError(index)
            self.levels_list[index] = level
        elif self.env == 'ant':
            if index == 0:
                level = AntEnv(f'antmaze-umaze-v0', reward_type=reward_type, **kwargs)  # 0
            elif index == 1:
                level = AntEnv(f'antmaze-umaze-diverse-v0', reward_type=reward_type, **kwargs)  # 1
            elif index == 2:
                level = AntEnv(f'antmaze-medium-diverse-v0', reward_type=reward_type, **kwargs)  # 2
            elif index == 3:
                level = AntEnv(f'antmaze-large-diverse-v0', reward_type=reward_type, **kwargs)  # 3
            elif index == 4:
                level = AntEnv(f'antmaze-open-v0', reward_type=reward_type, **kwargs)  # 4
            elif index == 5:
                level = AntEnv(f'antmaze-umaze-easy-v0', reward_type=reward_type, **kwargs)  # 5
            elif index == 6:
                level = AntEnv('antmaze-randommaze-v0', reward_type=reward_type, **kwargs)  # 6
            elif index == 7:
                level = AntEnv('antmaze-randommaze-small-v0', reward_type=reward_type, **kwargs)  # 7
            elif index == 8:
                level = AntEnv('antmaze-randommaze-medium-v0', reward_type=reward_type, **kwargs)  # 8
            elif index == 9:
                level = AntEnv('antmaze-randommaze-large-v0', reward_type=reward_type, **kwargs)  # 9
            elif index == 10:
                level = AntEnv('antmaze-randommaze-huge-v0', reward_type=reward_type, **kwargs)  # 10
            elif index == 11:
                level = AntEnv('antmaze-6x6-v0', reward_type=reward_type, **kwargs)  # 11
            else:
                raise NotImplementedError(index)
            level.seed(seed)
            self.levels_list[index] = level
        else:
            raise NotImplementedError(self.env)
        self._wrapped_env = level

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        # """
        try:
            if attr == '__len__':
                return None
            results = self.__getattribute__(attr)
            return results
        except:
            if hasattr(self._wrapped_env, '_wrapped_env'):
                orig_attr = self._wrapped_env.__getattr__(attr)
            else:
                orig_attr = self._wrapped_env.__getattribute__(attr)

            if callable(orig_attr):
                def hooked(*args, **kwargs):
                    result = orig_attr(*args, **kwargs)
                    return result

                return hooked
            else:
                return orig_attr

    def update_distribution_from_other(self, other):
        self.distribution = other.distribution.copy()
        self.index = other.index

    def advance_curriculum(self, index=None):
        if index is None:
            index = self.index + 1
        if index >= len(self.levels_list):
            print("LEARNED ALL THE LEVELS!!")
            raise NotImplementedError("Invalid level")
        if self.advance_curriculum_func == 'one_hot':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[index] = 1
        elif self.advance_curriculum_func == 'smooth':
            # Advance curriculum by assigning 0.9 probability to the new environment and 0.1 to all past environments.
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[index] = 0.9
            num_past_levels = index
            prev_env_prob = 0.1 / num_past_levels
            self.distribution[:index] = prev_env_prob
        elif self.advance_curriculum_func == 'four_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[16, 22, 23, 24]] = .25
        elif self.advance_curriculum_func == 'four_big_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[22, 23, 24, 25]] = .25
        elif self.advance_curriculum_func == 'five_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[16, 22, 23, 24, 25]] = .2
        elif self.advance_curriculum_func == 'goto_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[9, 14, 23]] = 1 / 3.
        elif self.advance_curriculum_func == 'easy_goto':
            self.distribution = np.zeros((len(self.levels_list)))
            indices = [4, 9, 14, 7, 12, 17, 39, 40, 41, 42, 43, 44, 46, 47]
            self.distribution[indices] = 1. / len(indices)
        elif self.advance_curriculum_func == 'uniform':
            # uniform probability over all envs we've seen so far
            self.distribution = np.zeros((len(self.levels_list)))
            prob = 0.1 / (index + 1)
            self.distribution[:index + 1] = prob
        else:
            raise ValueError('invalid curriculum type' + str(self.advance_curriculum_func))
        self.index = index
        print("updated curriculum", self.index, type(self.levels_list[self.index]))

    def set_level(self, index):
        """
        Set the curriculum at a certain level
        :param index: Index of the level to use
        """
        self.set_wrapped_env(index)

    def set_level_distribution(self, index):
        """
        Set the curriculum at a certain level, and set the distribution to only sample that level.
        :param index: Index of the level to use
        """
        self.set_wrapped_env(index)
        self.distribution = np.zeros((len(self.levels_list)))
        self.distribution[index] = 1
        self.index = index

    def seed(self, i):
        levels_list = self.levels_list if type(self.levels_list) is list else self.levels_list.values()
        if type(self.levels_list) is list:
            for level in self.levels_list:
                level.seed(int(i))
        elif type(self.levels_list) is dict:
            for k, level in self.levels_list.items():
                if type(level) is int:
                    self.levels_list[k] = i
                else:
                    level.seed(int(i))

    def set_task(self, args=None):
        """
        Each time we set a task, sample which babyai level to use from the categorical distribution array.
        Then set the task as usual.
        """
        env_index = self.np_random.choice(np.arange(len(self.distribution)), p=self.distribution)
        self.set_wrapped_env(env_index)
        return self._wrapped_env.set_task(args)

    def copy(self, index=None):
        env = copy.deepcopy(self)
        if index is None:
            index = self.index
        env.set_level_distribution(index)
        env.set_task()
        env.reset()
        return env
