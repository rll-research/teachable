from meta_mb.utils.serializable import Serializable
import numpy as np
from gym.spaces import Box


class ImgWrapperEnv(Serializable):
    def __init__(self, env, vae=None,
                 use_img=True, img_size=(64, 64, 3),
                 latent_dim=None, time_steps=4):

        Serializable.quick_init(self, locals())
        assert len(img_size) == 3

        self._wrapped_env = env
        self._vae = vae
        self._use_img = use_img
        self._img_size = img_size
        self._num_chan = img_size[-1]
        self._latent_dim = latent_dim
        self._time_steps = time_steps

    def step(self, action):
        _, reward, done, info = self._wrapped_env.step(action)
        obs = self.render('rgb_array', width=self._img_size[0], height=self._img_size[1]) / 255.
        self._obs[:, :, self._num_chan:] = self._obs[:, :, :-self._num_chan]
        self._obs[:, :, :self._num_chan] = obs

        if self._vae is not None:
            obs = self._vae.encode(self._obs).reshape((self._latent_dim,))
        else:
            obs = self._obs

        return obs, reward, done, info

    def reset(self):
        _ = self._wrapped_env.reset()
        self._obs = np.zeros(self._img_size[:-1] + (self._num_chan * self._time_steps,))
        obs = self.render('rgb_array', width=self._img_size[0], height=self._img_size[1]) / 255.
        self._obs[:, :, :self._num_chan] = obs

        if self._vae is not None:
            obs = self._vae.encode(self._obs).reshape((self._latent_dim,))
        else:
            obs = self._obs

        return obs

    @property
    def observation_space(self):
        if self._latent_dim is not None:
            assert self._use_img
            return Box(-1e6 * np.ones((self._latent_dim,)),
                       1e6 * np.ones((self._latent_dim,)), dtype=np.float32)

        return Box(-1e6 * np.ones(self._img_size + (self._n_channels,)),
                   1e6 * np.ones(self._img_size + (self._n_channels,)),
                   dtype=np.float32)

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        # orig_attr = self._wrapped_env.__getattribute__(attr)
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


image_wrapper = ImgWrapperEnv