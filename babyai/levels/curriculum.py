from meta_mb.utils.serializable import Serializable
from babyai.levels.iclr19_levels import *


class Curriculum(Serializable):
    def __init__(self, advance_curriculum_func, start_index=None, **kwargs):
        """

        :param advance_curriculum_func: Either 'one_hot' or 'smooth' depending on whether you want each level of the
        curriculum to be a single environment or a distribution over past environments
        :param start_index: what index of the curriculum to start on
        :param kwargs: arguments for the environment
        """
        Serializable.quick_init(self, locals())
        self.advance_curriculum_func = advance_curriculum_func
        # List of all the levels.  There are actually a bunch more: some ones which were omitted since they were
        # very similar to the current ones (e.g. more Level_GoToLocal variants with different sizes and num dists)
        # also some harder levels with multiple instructions chained together.
        self.levels_list = [
            Level_GoToRedBallNoDists(**kwargs),  # 0
            # Level_GoToRedBallGrey(**kwargs),  # 1
            # Level_GoToRedBall(**kwargs),  # 2
            # Level_GoToObjS4(**kwargs),  # 3
            # Level_GoToObjS6(**kwargs),  # 4
            # Level_GoToObj(**kwargs),  # 5
            # Level_GoToLocalS5N2(**kwargs),  # 6
            # Level_GoToLocalS6N3(**kwargs),  # 7
            # Level_GoToLocalS7N4(**kwargs),  # 8
            # Level_GoToLocalS8N7(**kwargs),  # 9
            # Level_GoToLocal(**kwargs),  # 10
            # Level_PickupLocalS5N2(**kwargs),  # 11
            # Level_PickupLocalS6N3(**kwargs),  # 12
            # Level_PickupLocalS7N4(**kwargs),  # 13
            # Level_PickupLocalS8N7(**kwargs),  # 14
            # Level_PickupLocal(**kwargs), #  15
            # Level_PutNextLocalS5N3(**kwargs), # 16
            # Level_PutNextLocalS6N4(**kwargs),  # 17
            # Level_PutNextLocal(**kwargs),  # 18
            # Level_OpenLocalS5N3(**kwargs),  # 19
            # Level_OpenLocalS6N4(**kwargs),  # 20
            # Level_OpenLocal(**kwargs),  # 21
            # Level_GoToObjMazeOpen(**kwargs),  # 22
            # Level_GoToOpen(**kwargs),  # 23
            # Level_GoToObjMazeS4R2(**kwargs),  # 24
            # Level_GoToObjMazeS5(**kwargs),  # 25
            # Level_GoToObjMaze(**kwargs),  # 26
            # Level_Open(**kwargs),  # 27
            # Level_GoTo(**kwargs),  # 28
            # Level_Pickup(**kwargs),  # 29
            # Level_Unlock(**kwargs),  # 30
            # Level_GoToImpUnlock(**kwargs),  # 31
            # Level_PutNext(**kwargs),  # 32
            # Level_UnblockPickup(**kwargs),  # 33
        ]
        # If start index isn't specified, start from the beginning (if we're using the pre-levels), or start
        # from the end of the pre-levels.
        if start_index is None:
            start_index = 0
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
            num_past_levels = index
            prev_env_prob = 0.1 / num_past_levels
            self.distribution[:index] = prev_env_prob
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
