from setuptools import Command, find_packages, setup
import os

NAME = "scoresbibm"

entry_points={
    'console_scripts': [
        'scoresbi = main.script:main',
    ],
}

os.system("pip install sbibm --no-deps")

REQUIRED = [
    "numpy",
    "matplotlib",
    "jax",
    "torch",
    "hydra-core",
    "hydra-submitit-launcher",
    "hydra-optuna-sweeper",
    "omegaconf",
    "sbi",
    "optuna",
]

setup(name=NAME,install_requires=REQUIRED,packages=find_packages("main"), entry_points=entry_points)