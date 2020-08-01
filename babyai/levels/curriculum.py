from meta_mb.utils.serializable import Serializable
from babyai.levels.iclr19_levels import *


class Curriculum(Serializable):
    def __init__(self, advance_curriculum_func, start_index=None, pre_levels=False, **kwargs):
        """

        :param advance_curriculum_func: Either 'one_hot' or 'smooth' depending on whether you want each level of the
        curriculum to be a single environment or a distribution over past environments
        :param start_index: what index of the curriculum to start on
        :param pre_levels: Whether or not to first run the 'pre_levels' which are designed to just introduce the
        different actions the agent can take and the teacher's feedback
        :param kwargs: arguments for the environment
        """
        Serializable.quick_init(self, locals())
        self.advance_curriculum_func = advance_curriculum_func
        self.pre_levels = pre_levels
        # List of all the levels.  There are actually a bunch more: some ones which were omitted since they were
        # very similar to the current ones (e.g. more Level_GoToLocal variants with different sizes and num dists)
        # also some harder levels with multiple instructions chained together.
        self.pre_levels_list = [
            Level_IntroPrimitivesD0Strict(**kwargs),
            Level_IntroPrimitivesD1Strict(**kwargs),
            Level_IntroPrimitivesD5Strict(**kwargs),
            Level_IntroPrimitivesD10Strict(**kwargs),
        ]
        # self.pre_levels_list = [
        #     Level_IntroPrimitivesD0(**kwargs),
        #     Level_IntroPrimitivesD1(**kwargs),
        #     Level_IntroPrimitivesD5(**kwargs),
        #     Level_IntroPrimitivesD10(**kwargs),
        # ]
        self.normal_levels_list = [
            Level_GoToRedBallNoDists(**kwargs),
            # Level_GoToRedBallGrey(**kwargs),
            # Level_GoToRedBall(**kwargs),
            # Level_GoToObjS4(**kwargs),
            # Level_GoToObjS6(**kwargs),
            # Level_GoToObj(**kwargs),
            # Level_GoToLocalS5N2(**kwargs),
            # Level_GoToLocalS6N3(**kwargs),
            # Level_GoToLocalS7N4(**kwargs),
            # Level_GoToLocalS8N7(**kwargs),
            # Level_GoToLocal(**kwargs),
            # Level_PickupLocalS5N2(**kwargs), # hard even with action teacher
            # Level_PickupLocalS6N3(**kwargs),
            # Level_PickupLocalS7N4(**kwargs),
            # Level_PickupLocalS8N7(**kwargs),
            # Level_PickupLocal(**kwargs), # hard with postaction teacher
            # Level_PutNextLocalS5N3(**kwargs), # hard even with preaction teacher
            # Level_PutNextLocalS6N4(**kwargs),
            # Level_PutNextLocal(**kwargs),
            # Level_OpenLocalS5N3(**kwargs),
            # Level_OpenLocalS6N4(**kwargs),
            # Level_OpenLocal(**kwargs),
            # Level_GoToObjMazeOpen(**kwargs),
            # Level_GoToOpen(**kwargs),
            # Level_GoToObjMazeS4R2(**kwargs),
            # Level_GoToObjMazeS5(**kwargs),
            # Level_GoToObjMaze(**kwargs),
            # Level_Open(**kwargs),
            # Level_GoTo(**kwargs),
            # Level_Pickup(**kwargs),
            # Level_Unlock(**kwargs),
            # Level_GoToImpUnlock(**kwargs),
            # Level_PutNext(**kwargs),
            # Level_UnblockPickup(**kwargs),
        ]
        # If start index isn't specified, start from the beginning (if we're using the pre-levels), or start
        # from the end of the pre-levels.
        if start_index is None:
            if pre_levels:
                start_index = 0
            else:
                start_index = len(self.pre_levels_list)
        self.levels_list = self.pre_levels_list + self.normal_levels_list
        self.distribution = np.zeros((len(self.levels_list)))
        self.distribution[start_index] = 1
        self._wrapped_env = self.levels_list[start_index]
        self.index = start_index


    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        # """
        try:
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

    def advance_curriculum(self, index=None):
        if index is None:
            index = self.index + 1
        if self.advance_curriculum_func == 'one_hot':
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[index] = 1
        elif self.advance_curriculum_func == 'smooth':
            # Advance curriculum by assigning 0.9 probability to the new environment and 0.1 to all past environments.
            self.distribution = np.zeros((len(self.levels_list)))
            self.distribution[index] = 0.9
            if self.pre_levels:
                prev_env_prob = 0.1 / index
                self.distribution[:index] = prev_env_prob
            else:
                num_past_levels = index if self.pre_levels else index - len(self.pre_levels_list)
                prev_env_prob = 0.1 / num_past_levels
                self.distribution[len(self.pre_levels_list):index] = prev_env_prob
        else:
            raise ValueError('invalid curriculum type' + str(self.advance_curriculum_func))
        self.index = index
        if self.index >= len(self.levels_list):
            print("LEARNED ALL THE LEVELS!!")
        print("updated curriculum", self.index, type(self.levels_list[self.index]))

    def set_level(self, index):
        """
        Set the curriculum at a certain level
        :param index: Index of the level to use
        """
        self._wrapped_env = self.levels_list[index]

    def set_level_distribution(self, index):
        """
        Set the curriculum at a certain level, and set the distribution to only sample that level.
        :param index: Index of the level to use
        """
        self._wrapped_env = self.levels_list[index]
        self.distribution = np.zeros((len(self.levels_list)))
        self.distribution[index] = 1
        self.index = index

    def set_task(self, args):
        """
        Each time we set a task, sample which babyai level to use from the categorical distribution array.
        Then set the task as usual.
        """
        env_index = np.random.choice(np.arange(len(self.distribution)), p=self.distribution)
        self._wrapped_env = self.levels_list[env_index]
        return self._wrapped_env.set_task(args)
