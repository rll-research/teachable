from envs.babyai.levels.iclr19_levels import *
from utils.serializable import Serializable
try:
    from envs.d4rl.d4rl_content.locomotion import *
    from envs.d4rl_envs import PointMassEnv, AntEnv
    from envs.dummy_envs import PointMassEnvSimple, DummyDiscrete
except Exception as e:
    print("Unable to load AntMaze, likely because Mujoco isn't properly installed.  This is fine so long as you only use BabyAI.")
import copy
NULL_SEED = 1000

class EnvDist(Serializable):
    def __init__(self, env_dist, env, start_index=0, reward_type='dense', **kwargs):
        """
        :param start_index: what env to start on
        :param kwargs: arguments for the environment
        """
        Serializable.quick_init(self, locals())
        reward_env_name = '' if reward_type == 'sparse' else '-dense'
        self.env_dist = env_dist
        self.env = env
        self.kwargs = kwargs
        self.reward_type = reward_type
        self.reward_env_name = reward_env_name
        if env == 'dummy':
            self.levels_list = {0: NULL_SEED}
        if env == 'dummy_discrete':
            self.levels_list = {0: NULL_SEED}
        if env == 'point_mass':
            self.levels_list = {k: NULL_SEED for k in range(15)}
        elif env == 'ant':
            self.levels_list = {k: NULL_SEED for k in range(12 + 10)}
        elif env == 'babyai':
            self.levels_list = {k: NULL_SEED for k in range(56)}
        # If start index isn't specified, start from the beginning (if we're using the pre-levels), or start
        # from the end of the pre-levels.
        if self.env_dist == 'four_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[16, 22, 23, 24]] = .25
        elif self.env_dist == 'four_big_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[22, 23, 24, 25]] = .25
        elif self.env_dist == 'five_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[16, 22, 23, 24, 25]] = .2
        elif self.env_dist == 'goto_levels':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[[9, 14, 23]] = 1 / 3.
        elif self.env_dist == 'easy_goto':
            self.distribution = np.zeros((len(self.levels_list)))
            indices = [4, 9, 14, 7, 12, 17, 39, 40, 41, 42, 43, 44, 46, 47]
            self.distribution[indices] = 1. / len(indices)
        elif env_dist == 'uniform':
            prob_mass = 1 / (start_index + 1)
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[:start_index + 1] = prob_mass
        else:
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[start_index] = 1
        self.set_wrapped_env(start_index)
        self.index = start_index

    def set_wrapped_env(self, index):
        if not type(self.levels_list[index]) is int:
            self._wrapped_env = self.levels_list[index]
            return
        kwargs = self.kwargs
        reward_env_name = self.reward_env_name
        reward_type = self.reward_type
        seed = self.levels_list[index]
        if self.env == 'dummy':
            level = PointMassEnvSimple(**kwargs)
        elif self.env == 'dummy_discrete':
            level = DummyDiscrete(**kwargs)
        elif self.env == 'point_mass':
            if index == 0:
                level = PointMassEnv(f'maze2d-open{reward_env_name}-v0', reward_type=reward_type, **kwargs)  # 0
            elif index == 1:
                level = PointMassEnv(f'maze2d-umaze{reward_env_name}-v1', reward_type=reward_type, **kwargs)  # 1
            elif index == 2:
                level = PointMassEnv(f'maze2d-medium{reward_env_name}-v1', reward_type=reward_type, **kwargs)  # 2
            elif index == 3:
                level = PointMassEnv(f'maze2d-large{reward_env_name}-v1', reward_type=reward_type, **kwargs)  # 3
            elif index == 4:
                level = PointMassEnv('maze2d-randommaze-v0', reward_type=reward_type, **kwargs)  # 4
            elif index == 5:
                level = PointMassEnv(f'maze2d-umaze{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 5
                             **kwargs)
            elif index == 6:
                level = PointMassEnv(f'maze2d-medium{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 6
                             **kwargs)
            elif index == 7:
                level = PointMassEnv(f'maze2d-large{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 7
                             **kwargs)
            elif index == 8:
                level = PointMassEnv(f'maze2d-umaze{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 8
                             reset_start=False, **kwargs)
            elif index == 9:
                level = PointMassEnv(f'maze2d-medium{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 9
                             reset_start=False, **kwargs)
            elif index == 10:
                level = PointMassEnv(f'maze2d-large{reward_env_name}-v1', reward_type=reward_type, reset_target=False,  # 10
                             reset_start=False, **kwargs)
            elif index == 11:
                level = PointMassEnv('maze2d-randommaze-7x7-v0', reward_type=reward_type, **kwargs)  # 11
            elif index == 12:
                level = PointMassEnv('maze2d-randommaze-8x8-v0', reward_type=reward_type, **kwargs)  # 12
            elif index == 13:
                level = PointMassEnv('maze2d-12x12-v0', reward_type=reward_type, **kwargs)  # 13
            elif index == 14:
                level = PointMassEnv('maze2d-15x15-v0', reward_type=reward_type, **kwargs)  # 14
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
            elif index >= 12 and index < 22:
                i = index - 12
                level = AntEnv(f'antmaze-fixed{i}-6x6-v0', reward_type=reward_type, **kwargs)
            else:
                raise NotImplementedError(index)
            level.seed(seed)
            self.levels_list[index] = level
        elif self.env == 'babyai':
            # Easy levels
            if index == 0:
                level = Level_GoToRedBallNoDists(**kwargs) # 0 --> intro L, R, Forward PreAction, Explore and GoNextTo subgoals
            elif index == 1:
                level = Level_GoToRedBallGrey(**kwargs)  # 1 --> first level with distractors
            elif index == 2:
                level = Level_GoToRedBall(**kwargs)  # 2 --> first level with colored distractors
            elif index == 3:
                level = Level_GoToObjS5(**kwargs)  # 3 --> first level where the goal is something other than a red ball
            elif index == 4:
                level = Level_GoToLocalS5N2(**kwargs)  # 4 --> first level where the task means something
            elif index == 5:
                level = Level_PickupLocalS5N2(**kwargs)  # 5 --> intro Pickup subgoal and pickup PreAction
            elif index == 6:
                level = Level_PutNextLocalS5N2(**kwargs)  # 6 --> intro Drop subgoal and drop PreAction
            elif index == 7:
                level = Level_OpenLocalS5N2(**kwargs)  # 7 --> intro Open subgoal and open PreAction
            # Medium levels (here we introduce the harder teacher; no new tasks, just larger sizes)
            elif index == 8:
                level = Level_GoToObjS7(**kwargs)  # 8
            elif index == 9:
                level = Level_GoToLocalS7N4(**kwargs)  # 9
            elif index == 10:
                level = Level_PickupLocalS7N4(**kwargs)  # 10
            elif index == 11:
                level = Level_PutNextLocalS7N4(**kwargs)  # 11
            elif index == 12:
                level = Level_OpenLocalS7N4(**kwargs)  # 12
            # Hard levels (bigger sizes, some new tasks)
            elif index == 13:
                level = Level_GoToObj(**kwargs)  # 13
            elif index == 14:
                level = Level_GoToLocal(**kwargs)  # 14
            elif index == 15:
                level = Level_PickupLocal(**kwargs)  # 15
            elif index == 16:
                level = Level_PutNextLocal(**kwargs)  # 16
            elif index == 17:
                level = Level_OpenLocal(**kwargs)  # 17
            # Biggest levels (larger grid)
            elif index == 18:
                level = Level_GoToObjMazeOpen(**kwargs)  # 18
            elif index == 19:
                level = Level_GoToOpen(**kwargs)  # 19
            elif index == 20:
                level = Level_GoToObjMazeS4R2(**kwargs)  # 20
            elif index == 21:
                level = Level_GoToObjMazeS5(**kwargs)  # 21
            elif index == 22:
                level = Level_Open(**kwargs)  # 22
            elif index == 23:
                level = Level_GoTo(**kwargs)  # 23
            elif index == 24:
                level = Level_Pickup(**kwargs)  # 24
            elif index == 25:
                level = Level_PutNext(**kwargs)  # 25
            elif index == 26:
                # Larger sizes than we've seen before
                level = Level_PickupObjBigger(**kwargs)  # 26 test0
            elif index == 27:
                # More distractors than we've seen before
                level = Level_GoToObjDistractors(**kwargs)  # 27 test1
            elif index == 28:
                # New object
                level = Level_GoToHeldout(**kwargs)  # 28 test2
            elif index == 29:
                # Task we've seen before, but new instructions
                level = Level_GoToGreenBox(**kwargs)  # 29 test3
            elif index == 30:
                level = Level_PutNextSameColor(**kwargs)  # 30 test4
            elif index == 31:
                # New object
                level = Level_Unlock(**kwargs)  # 31 test5 ("unlock" is a completely new instruction)
            elif index == 32:
                level = Level_GoToImpUnlock(**kwargs)  # 32 test6
            elif index == 33:
                level = Level_UnblockPickup(**kwargs)  # 33 test7 (known task, but now there's the extra step of unblocking)
            elif index == 34:
                level = Level_Seek(**kwargs)  # 34 test8
            elif index == 35:
                # Easier heldout levels
                level = Level_GoToGreenBoxLocal(**kwargs)  # 35 test9
            elif index == 36:
                level = Level_PutNextSameColorLocal(**kwargs)  # 36 test10
            elif index == 37:
                level = Level_UnlockLocal(**kwargs)  # 37 test11 ("unlock" is a completely new instruction)
            elif index == 38:
                level = Level_GoToImpUnlockLocal(**kwargs)  # 38 test12
            elif index == 39:
                level = Level_SeekLocal(**kwargs)  # 39 test13
            elif index == 40:
                level = Level_GoToObjDistractorsLocal(**kwargs)  # 40 test14
            elif index == 41:
                level = Level_GoToSmall2by2(**kwargs)  # 41 test15
            elif index == 42:
                level = Level_GoToSmall3by3(**kwargs)  # 42 test16
            elif index == 43:
                level = Level_SeekSmall2by2(**kwargs)  # 43 test17
            elif index == 44:
                level = Level_SeekSmall3by3(**kwargs)  # 44 test18
            elif index == 45:
                level = Level_GoToObjDistractorsLocalBig(**kwargs)  # 45 test19
            elif index == 46:
                level = Level_OpenSmall2by2(**kwargs)  # 46 test20
            elif index == 47:
                level = Level_OpenSmall3by3(**kwargs)  # 47 test21
            elif index == 48:
                level = Level_SeekL0(**kwargs)  # 48 test22
            elif index == 49:
                level = Level_UnlockTopLeft(**kwargs)  # UnlockTopLeft
            elif index == 50:
                level = Level_UnlockTopLeftRed(**kwargs)  # UnlockTopLeft, only red door/key
            elif index == 51:
                level = Level_UnlockTopLeftFixedStart(**kwargs)  # Agent always starts in same place
            elif index == 52:
                level = Level_UnlockTopLeftFixedDoor(**kwargs)  # Agent always ends in same place
            elif index == 53:
                level = Level_UnlockTopLeftFixedKey(**kwargs)  # Key is always in same place
            elif index == 54:
                level = Level_UnlockTopLeftFixedKeyDoor(**kwargs)  # Key and door are always in same place
            elif index == 55:
                level = Level_UnlockTopLeftFixedAll(**kwargs)  # Agent, key and door are always in same place

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

    def set_level(self, index):
        """
        :param index: Index of the level to use
        """
        self.set_wrapped_env(index)

    def set_level_distribution(self, index=None, copy_distribution=None):
        """
        Choose a particular level, and set the distribution to only sample that level.
        :param index: Index of the level to use
        """
        self.set_wrapped_env(index)
        if copy_distribution is not None:
            self.distribution = copy_distribution
        else:
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[index] = 1
        self.index = index

    def seed(self, i):
        if type(self.levels_list) is list:
            for level in self.levels_list:
                level.seed(int(i))
        elif type(self.levels_list) is dict:
            for k, level in self.levels_list.items():
                if type(level) is int:
                    self.levels_list[k] = i
                else:
                    level.seed(int(i))

    def set_task(self):
        """
        Each time we set a task, sample which babyai level to use from the categorical distribution array.
        Then set the task as usual.
        """
        env_index = self.np_random.choice(np.arange(len(self.distribution)), p=self.distribution)
        self.set_wrapped_env(env_index)

    def reset(self):
        self.set_task()
        return self._wrapped_env.reset()

    def copy(self, index=None):
        env = copy.deepcopy(self)
        if index is None:
            index = self.index
        env.set_level_distribution(index=index, copy_distribution=self.distribution.copy())
        env.reset()
        return env
