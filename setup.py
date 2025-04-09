"""Setup script for belief_srs"""

from setuptools import setup

requirements = [
    "autolab_core>=1.1.0",
    'black',
    # 'd3rlpy',
    'gymnasium==0.28.1',
    # "gym==0.25.1",
    'hydra-core',
    'hydra-ray-launcher',
    'hydra-joblib-launcher',
    'hydra-colorlog',
    'ipdb',
    'matplotlib',
    'minigrid',
    'numpy==1.26.4',
    'opencv-python>=3',
    'plotly',
    'pyquaternion',
    'pytest',
    'ray',
    'rl_utils@https://github.com/iamlab-cmu/rl-utils/archive/refs/tags/1.0.0.zip',
    'scikit-learn',
    'robosuite==1.4.0',
    'mujoco>=2.3.0',
    'scikit-spatial',
    'stable-baselines3[extra]==2.2.1',
    'termcolor',
    # 'torch==2.1.0+cu118" # install torch manually
    # tensordict-nightly
    # torchrl # clone from github and install

    # for torchrl
    # there is come incompatibility b/w gym versions for torchrl and other packages
    # "torchrl==0.1.1",
    # "tensordict==0.1.2",
    # "charset-normalizer-3.1.0",
    #----
    'tqdm',
    'wandb',
]

setup(
    name="belief_srs",
    version="0.1.0",
    author="Shivam Vats",
    author_email="svats@cs.cmu.edu",
    packages=["belief_srs"],
    install_requires=requirements,
)
