from distutils.core import setup
from setuptools import find_packages

setup(
    name='d4rl_content',
    version='1.1',
    install_requires=['gym', 
                      'numpy', 
                      'mujoco_py', 
                      'pybullet',
                      'h5py',], 
    packages=find_packages(),
    package_data={'d4rl_content': ['locomotion/assets/*',
                           ]},
    include_package_data=True,
)
