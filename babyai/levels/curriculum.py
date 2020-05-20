from meta_mb.utils.serializable import Serializable
from babyai.levels.iclr19_levels import *


class Curriculum(Serializable):
    def __init__(self, advance_curriculum_func, **kwargs):
        Serializable.quick_init(self, locals())
        # List of all the levels.  There are actually a bunch more: some ones which were omitted since they were
        # very similar to the current ones (e.g. more Level_GoToLocal variants with different sizes and num dists)
        # also some harder levels with multiple instructions chained together.
        self.levels_list = [Level_GoToRedBallNoDists(**kwargs),
                            Level_GoToRedBallGrey(**kwargs),
                            Level_GoToRedBall(**kwargs),
                            Level_GoToObjS4(**kwargs),
                            Level_GoToObjS6(**kwargs),
                            Level_GoToObj(**kwargs),
                            Level_GoToLocalS5N2(**kwargs),
                            Level_GoToLocalS6N3(**kwargs),
                            Level_GoToLocalS7N4(**kwargs),
                            Level_GoToLocalS8N7(**kwargs),
                            Level_GoToLocal(**kwargs),
                            Level_PutNextLocalS5N3(**kwargs),
                            Level_PutNextLocalS6N4(**kwargs),
                            Level_PutNextLocal(**kwargs),
                            Level_PutNext(**kwargs),
                            Level_GoToObjMazeOpen(**kwargs),
                            Level_GoToOpen(**kwargs),
                            Level_GoToObjMazeS4R2(**kwargs),
                            Level_GoToObjMazeS5(**kwargs),
                            Level_GoToObjMaze(**kwargs),
                            Level_Open(**kwargs),
                            Level_GoTo(**kwargs),
                            Level_Pickup(**kwargs),
                            Level_Unlock(**kwargs),
                            Level_GoToImpUnlock(**kwargs),
                            Level_UnblockPickup(**kwargs),
                            ]
        self.distribution = np.zeros((len(self.levels_list)))
        self.distribution[0] = 1
        self._wrapped_env = self.levels_list[0]
        self.index = 1
        # I tried this, but for whatever reason it broke curriculum updating # TODO: figure out what's up here
        # self.advance_curriculum = self.__getattr__(advance_curriculum_func)


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

    def advance_curriculum(self):
        """Advance the curriculum to 100% sample from the next hardest level. """
        curr_index = np.argmax(self.distribution)
        self.distribution = np.zeros((len(self.levels_list)))
        self.distribution[curr_index + 1] = 1
        self._wrapped_env = self.levels_list[self.index]
        self.index += 1
        if self.index > len(self.levels_list):
            print("LEARNED ALL THE LEVELS!!")
        print("updated curriculum", self.index - 1, type(self._wrapped_env))

    # def advance_curriculum(self):  # advance_curriculum_uniform_smooth
    #     """Advance curriculum by assigning 0.5 probability to the new environment and 0.5 to all past environments."""
    #     curr_index = np.argmax(self.distribution)
    #     self.distribution = np.zeros((len(self.levels_list)))
    #     self.distribution[curr_index + 1] = 0.5
    #     prev_env_prob = 0.5/(curr_index + 1)
    #     self.distribution[:curr_index + 1] = prev_env_prob
    #     print("updated curriculum", curr_index + 1, self.index - 1, type(self.levels_list[self.index]))
    #     self.index += 1
    #     if self.index > len(self.levels_list):
    #         print("LEARNED ALL THE LEVELS!!")

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

    def set_task(self, args):
        """
        Each time we set a task, sample which babyai level to use from the categorical distribution array.
        Then set the task as usual.
        """
        env_index = np.random.choice(np.arange(len(self.distribution)), p=self.distribution)
        self._wrapped_env = self.levels_list[env_index]
        return self._wrapped_env.set_task(args)
