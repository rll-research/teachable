import numpy as np
import zmq
from meta_mb.logger import logger
import gym
from gym import spaces
from meta_mb.meta_envs.base import MetaEnv
import time


class PR2Env(MetaEnv, gym.utils.EzPickle):
    PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

    def __init__(self):
        # self.goal = np.array([-0.1511672, 0.43030036, 0.71051866])
        # self.goal = np.array([-7.29517469e-02, -2.86581420e-02, 5.70482330e-01, -8.47117285e-02,
        #                       -1.18948075e-02, 5.98804157e-01, -5.13613156e-02, -8.77137857e-02,
        #                       5.85055245e-01])
        # self.goal = np.array([0.1644276, -0.31147851, 1.52381236,
        #                        -0.90102611, -4.98011356, -1.66494068, -1.01992367])
        self.goal = [np.array([ 5.96785857e-01, -2.85932172e-01,  1.59162625e+00, -1.10704422e+00,
 -5.06300837e+00, -1.71918953e+00, -6.13503858e-01, 2.79299305e-01,  3.57783994e-01,
  1.16489066e-01])
                     ]
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        print("Connecting to the server...")
        self.socket.connect("tcp://127.0.0.1:7777")
        max_torques = np.array([5] * 7)
        self.frame_skip = 4
        # self.init_qpos = np.array([0.8, 0.4, 1.5, -1., -1.7, -.5, 0.])
        # self.init_qpos = np.array([0.7783511, -0.25606883, 1.12741523,
        #                            -0.87482262, -7.52093116, -0.09122304, 3.15171159])
 #        self.init_qpos = np.array([7.10011717e-01, -3.56398411e-01,  9.63204825e-01, -9.12897313e-01,
 # -4.66548326e+00, -2.05669173e+00, -2.77487280e-01])
        # self.init_qpos = np.array([0.5, -0.5, 1, -0.5, -5, 0, 1])
        # self.init_qpos = np.array([0.7, -0.2, 1.1, -0.8, -7.5, 0, 3])
        # self.init_qpos = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.init_qpos = np.array([0,0,0,0,0,0,0])


        self.act_dim = 7
        self.obs_dim = 23 - 6
        self._t = 0
        # self.obs_dim = 4
        self._init_obs = self.reset(real=True).copy()
        self._low, self._high = -max_torques, max_torques
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        ob = self.do_simulation(action, self.frame_skip)
        # time.sleep(1 / 20)
        # reward_dist = -np.linalg.norm(ob[-3:] - self.goal)
        # reward_dist = -np.linalg.norm(ob - self.goal[self.idx])
        # reward_ctrl = -np.square(ob[7:14]).sum()
        reward_dist = np.exp(-np.linalg.norm(ob[:5] - self.goal[self.idx][:5]))
        reward_vel = .5 * np.exp(-np.linalg.norm(ob[7:14]))
        reward_gripper = 2 * np.exp(-np.linalg.norm(np.concatenate([ob[5:7], ob[-3:]], axis=-1)  - self.goal[self.idx][5:]))
        reward = reward_dist + reward_vel + reward_gripper
        done = False
        self._t += 1
        return ob, reward, done, dict(reward_dist=reward_dist) # , reward_ctrl=reward_ctrl)

    def do_simulation(self, action, frame_skip):
        assert frame_skip > 0
        if action.ndim == 2:
            action = action.reshape(-1)
        action = np.clip(action, self._low, self._high)
        # action = np.concatenate([[0] * 5, action])
        for _ in range(frame_skip):
            md = dict(
                dtype=str(action.dtype),
                cmd="action",
            )
            self.socket.send_json(md, 0 | zmq.SNDMORE)
            self.socket.send(action, 0, copy=True, track=False)
            ob = self._get_obs()
        return ob

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            # reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act/(2 * self._high)), axis=1)
            # reward_ctrl = -0.5 * 0.1 * np.sum(np.square(obs[:, 7:14]), axis=1)
            # reward_dist = -np.linalg.norm(obs_next[:,-3:] - self.goal, axis=1)
            reward_dist = np.exp(-np.linalg.norm(obs_next[:, :5] - self.goal[self.idx][:5], axis=1))
            reward_vel = .5 * np.exp(-np.linalg.norm(obs_next[:, 7:14], axis=1))
            reward_gripper = 2 * np.exp(-np.linalg.norm(np.concatenate([obs_next[:, 5:7], obs_next[:, -3:]],
                                                                       axis=-1) - self.goal[self.idx][5:], axis=1))
            reward = reward_dist + reward_vel + reward_gripper
            return np.clip(reward, -100, 100)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            # reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act/(2 * self._high)))
            # reward_ctrl = -0.5 * 0.1 * np.sum(np.square(obs[7:14]))
            # reward_dist = -np.linalg.norm(obs_next[-3:] - self.goal)
            reward_dist = np.exp(-np.linalg.norm(obs_next[:5] - self.goal[self.idx][:5]))
            reward_vel = .5 * np.exp(-np.linalg.norm(obs_next[7:14]))
            reward_gripper = 2 * np.exp(-np.linalg.norm(np.concatenate([obs_next[5:7], obs_next[-3:]], axis=-1) - self.goal[self.idx][5:]))
            reward = reward_dist + reward_vel + reward_gripper
            return np.clip(reward, -100, 100)
        else:
            raise NotImplementedError

    def reset(self, real=False):
        self._t = 0
        if real:
            print('real')
            qpos = self.init_qpos + np.random.uniform(-0.01, 0.01, size=len(self.init_qpos))
            md = dict(
                dtype=str(qpos.dtype),
                cmd="reset",

            )
            self.socket.send_json(md, 0 | zmq.SNDMORE)
            self.socket.send(qpos, 0, copy=True, track=False)
            return self._get_obs()
        else:
            return self._init_obs + np.random.uniform(-0.01, 0.01, size=len(self._init_obs))

    def _get_obs(self):
        msg = self.socket.recv(flags=0, copy=True, track=False)
        buf = memoryview(msg)
        obs = np.frombuffer(buf, dtype=np.float64)
        # return np.concatenate([obs.reshape(-1)[:2], obs.reshape(-1)[7:9]])
        return obs[:-6]

    def log_diagnostics(self, paths, prefix=''):
        dist = [-path["env_infos"]['reward_dist'] for path in paths]
        final_dist = [-path["env_infos"]['reward_dist'][-1] for path in paths]
        # ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDistance', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_dist))
        # logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))

    @property
    def action_space(self):
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def idx(self):
        return 0
        # if self._t < 10:
        #     return 0
        # else:
        #     return 1

if __name__ == "__main__":
    env = PR2Env()
    # print("reset!")
    # obs = env.reset()
    # obs = np.expand_dims(, axis=0)
    print(env._init_obs)
    # print("reset done!")
        # for _ in range(100):
        #     print("action!")
        #     a = env.action_space.sample() * 0
        #     env.step(a)
        #     env.reward(obs, np.expand_dims(a, axis=0), obs)
        #     print("action done!")

# Init:
#
# [ 5.42730494e-01  1.52270862e-02  9.43007182e-01 -8.68156264e-01
#  -5.32638623e+00 -1.53867780e+00  8.99776899e-01  2.31858976e-11
#  -6.93889390e-17  0.00000000e+00  8.80117496e-03  0.00000000e+00
#   0.00000000e+00  0.00000000e+00 -1.48569878e-01  2.43122203e-01
#   1.87660681e-01 -5.87167355e-02  2.69877152e-01  2.89092061e-01
#  -1.58908432e-01  2.16395001e-01  3.14027421e-01]
#
#  End:
# [ 1.42125719e-01 -1.45503268e-01  9.30820215e-01 -1.06374839e+00
#  -4.73241234e+00 -1.44477962e-01  1.58286694e+00  0.00000000e+00
#  -7.12379766e-09  0.00000000e+00  0.00000000e+00  0.00000000e+00
#   0.00000000e+00  0.00000000e+00 -7.29517469e-02 -2.86581420e-02
#   5.70482330e-01 -8.47117285e-02 -1.18948075e-02  5.98804157e-01
#  -5.13613156e-02 -8.77137857e-02  5.85055245e-01]


# Init from plate position
# [ 0.7783511  -0.25606883  1.12741523 -0.87482262 -7.52093116 -0.09122304
#   3.15171159  0.          0.          0.          0.          0.
#   0.          0.         -0.03861735  0.5429763   0.55299989 -0.02305524
#   0.59228718  0.64473468 -0.01265096  0.46321     0.64747213]


#PLATE position:
# 0.1644276  -0.31147851  1.52381236 -0.90102611 -4.98011356 -1.66494068
#  -1.01992367  0.          0.          0.          0.          0.
#   0.          0.         -0.22708802  0.07970474  0.15733524 -0.05736825
#   0.17155499  0.17208294 -0.10596121  0.02212625  0.24058344]


"""


Lego:
1. [ 0.74187219 -0.16986661  0.96786218 -0.76494165 -4.5891251  -1.94265812
  3.26905514  0.          0.          0.          0.          0.
  0.          0.         -0.16659146  0.49554746  0.21704888 -0.19025537
  0.58158067  0.20974212 -0.14368087  0.57942362  0.26686146]
  
2. [ 3.28085262e-01 -3.16469607e-01  1.18802936e+00 -9.21583556e-01
 -4.85452754e+00 -1.87099893e+00  2.64370143e+00  0.00000000e+00
 -1.11022302e-15  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.45424603e-01  1.37123813e-01
  2.56431151e-01 -1.69998881e-01  2.24088020e-01  2.55860864e-01
 -1.23184919e-01  2.21378144e-01  3.08828806e-01]
 
 
3. [0.12629056, -0.33922564,  1.25569909, -0.83081232, -4.90531728, -1.8157426,
  2.34105339,  0.,          0.,          0.,          0.,          0.,
  0.,          0.,         -0.14435956,  0.01431401,  0.25546218, -0.17143094,
  0.08309022,  0.26141724, -0.13745477,  0.09015565,  0.30613478]
  
  
4. [-0.06936906,-0.29151411,  1.52926443, -0.57891128, -4.95552855, -1.87387052,
  2.36106749,  0.,          0.,          0.,          0.,          0.,
  0.,          0.,         -0.13345914, -0.02537312,  0.16194666, -0.1599033,
  0.03485722, 0.18399872, -0.13409751, 0.04803587,  0.23454963]
  
5. [-1.43238858e-01, -2.00743753e-01, 1.39055750e+00, -3.84339446e-01,
 -4.77573980e+00, -1.93996057e+00,  2.49555356e+00,  0.00000000e+00,
 -1.83213444e-10,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00, -1.22261587e-01, -2.89328340e-02,
  1.21728281e-01, -1.47726911e-01,  3.27933283e-02,  1.51349782e-01,
 -1.21627484e-01,  4.52869719e-02,  1.98203254e-01]
"""

"""
[ 0.03492746 -0.44285442  1.51306859 -0.80359543 -5.16447223 -1.86651751
  2.31072767  0.          0.          0.          0.          0.
  0.          0.         -0.19274069 -0.03007735  0.2464475  -0.21953061
  0.0431868   0.26590113 -0.18416306  0.04940512  0.31146071]
  
  
  [ 0.06096014 -0.22290762  1.43930537 -0.87134812 -5.04669556 -1.87256525
  2.30224343  0.          0.          0.          0.          0.
  0.          0.         -0.1915515  -0.01845283  0.12999443 -0.21631334
  0.056541    0.16264397 -0.18166899  0.06205027  0.20291775]

"""


"""
Init: [ 6.08807068e-01, -3.93620177e-01,  1.25922689e+00, -9.09422816e-01,
 -4.91272171e+00, -1.93638719e+00,  2.68064616e+00,  0.00000000e+00,
  0.00000000e+00, -1.01669442e-05,  0.00000000e+00,  0.00000000e+00,
 -2.86612146e-02, -2.86612146e-02, -1.68403580e-01,  3.50214633e-01,
  2.73749418e-01, -1.95250427e-01,  4.13470716e-01,  2.77695352e-01,
 -1.65989693e-01,  4.25103612e-01,  3.34286505e-01]
 
 
Goal: [ 4.08587588e-01 -1.46010837e-01  1.28889254e+00 -8.87996750e-01
 -4.80784494e+00 -2.00043797e+00  2.48319703e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  8.88178420e-15 -1.51403786e-01  2.53451761e-01
  1.17586076e-01 -1.76419679e-01  3.10666054e-01  1.47617230e-01
 -1.53969955e-01  3.25107895e-01  1.94429943e-01]

"""


"""

Init (2): ([7.10011717e-01, -3.56398411e-01,  9.63204825e-01, -9.12897313e-01,
 -4.66548326e+00, -2.05669173e+00, -2.77487280e-01




Goal (2) : [ 5.96785857e-01, -2.85932172e-01,  1.59162625e+00, -1.10704422e+00,
 -5.06300837e+00, -1.71918953e+00, -6.13503858e-01, -1.58354262e-03,
  8.21743149e-03, -9.93711846e-02, -8.94682723e-03,  1.73478727e-02,
 -2.01818272e-02, -1.85441385e-02, -2.79299305e-01,  3.57783994e-01,
  1.16489066e-01, -1.07182981e-01,  4.33240193e-01,  1.20192246e-01,
 -1.68514010e-01,  2.89484657e-01,  1.87359024e-01]

"""




"""
obs for init_qpos=np.array([0,0,0,0,0,0,0])
  [ 0.09859975  0.09922984  0.80846948 -0.15038998  0.10019115 -0.09034941
  0.10817561  0.          0.          0.          0.          0.
  0.          0.          0.1448004   0.26479295  0.17570619]

  [ 0.48759759 -0.31427014  1.13511226 -0.5531421  -4.65912008 -1.99247238
  0.99332108  0.          0.          0.          0.          0.
  0.          0.         -0.16828345  0.40697539  0.26332118]


obs for init_qpos=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1])
  [-8.76432427e-03 -9.05165677e-03  8.67159349e-01 -1.51692912e-01
 -7.23088587e-03 -8.43016724e-02 -4.66166996e-04  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  1.39332611e-01  1.52058032e-01
  2.73096161e-01]

obs for init_qpos=np.array([0.5, -0.5, 1, -0.5, -5, 0, 1])
  [ 7.05061651e-01 -2.03366195e-01  1.12436849e+00 -8.07938547e-01
 -7.50843619e+00 -9.20897691e-02  3.00325915e+00  0.00000000e+00
  0.00000000e+00  2.61271893e-09  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -4.40685751e-02  4.99549120e-01
  5.37356806e-01]

obs for init_qpos=np.array([0.7, -0.2, 1.1, -0.8, -7.5, 0, 3])
  [ 0.46595897 -0.32120692  1.06359401 -0.51101382 -4.52624532 -1.9935601
  0.99031896  0.          0.          0.          0.          0.
  0.          0.         -0.16006932  0.41632504  0.27408067]




"""