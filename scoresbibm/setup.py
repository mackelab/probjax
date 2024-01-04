from setuptools import Command, find_packages, setup
import os

NAME = "scoresbibm"
VERSION = "0.0.1"
DESCRIPTION = "Score-based inference benchmark"
AUTHOR = "Anonymous"

entry_points={
    'console_scripts': [
        'scoresbi = src.scripts:main',
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
    "tueplots",
    "seaborn",
    "pandas",
]



setup(name=NAME,version=VERSION, description=DESCRIPTION, author=AUTHOR,package_dir={"scoresbibm": "src"}, install_requires=REQUIRED, entry_points=entry_points)