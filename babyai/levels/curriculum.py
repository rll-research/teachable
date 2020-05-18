from meta_mb.utils.serializable import Serializable
from babyai.levels.iclr19_levels import *


class Curriculum(Serializable):
    def __init__(self, advance_curriculum_func, **kwargs):
        Serializable.quick_init(self, locals())
        self.levels_list = [Level_GoToLocal(**kwargs),
                            Level_GoToRedBallGrey(**kwargs),
                            Level_GoToRedBallNoDists(**kwargs),
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
                            Level_Open(**kwargs),
                            Level_GoToOpen(**kwargs),
                            Level_GoToObjMazeOpen(**kwargs),
                            Level_GoToObjMazeS4R2(**kwargs),
                            Level_GoToObjMazeS5(**kwargs),
                            Level_GoToObjMaze(**kwargs),
                            Level_GoTo(**kwargs),
                            Level_Unlock(**kwargs),
                            Level_GoToImpUnlock(**kwargs),
                            Level_Pickup(**kwargs),
                            Level_PutNext(**kwargs),
                            Level_UnblockPickup(**kwargs),
                            ]
        self.distribution = np.zeros((len(self.levels_list)))
        self.distribution[0] = 1
        self._wrapped_env = self.levels_list[0]
        self.index = 1
        self.curr_score = 0
        self.past_scores = []
        self.advance_curriculum = self.__getattr__(advance_curriculum_func)


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

    def advance_curriculum_one_hot(self):
        """Currently we just advance one-by-one when this function is called.
        Later, it would be cool to advance dynamically when the agent has succeeded at a task.
        Also it would be ideal if we kept the past tasks around.
        """
        curr_index = np.argmax(self.distribution)
        self.distribution = np.zeros((len(self.levels_list)))
        self.distribution[curr_index + 1] = 1
        self._wrapped_env = self.levels_list[self.index]
        self.index += 1
        if self.index > len(self.levels_list):
            print("LEARNED ALL THE LEVELS!!")
        print("updated curriculum", type(self._wrapped_env))

    def advance_curriculum_uniform_smooth(self):
        """Advance curriculum by assigning 0.5 probability to the new environment and 0.5 to all past environments."""
        curr_index = np.argmax(self.distribution)
        self.distribution = np.zeros((len(self.levels_list)))
        self.distribution[curr_index + 1] = 0.5
        prev_env_prob = 0.5/(curr_index + 1)
        self.distribution[:curr_index + 1] = prev_env_prob
        print("updated curriculum", type(self.levels_list[self.index]))
        self.index += 1
        if self.index > len(self.levels_list):
            print("LEARNED ALL THE LEVELS!!")

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)
        self.curr_score += reward
        return obs, reward, done, info

    def reset(self):
        env_index = np.random.choice(np.arange(len(self.distribution)), p=self.distribution)
        self._wrapped_env = self.levels_list[env_index]
        return self._wrapped_env.reset()
