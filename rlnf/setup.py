from setuptools import setup

__version__ = "0.1.0"
requirements = open("requirements.txt", encoding="utf-8").read().splitlines()

setup(
    name="rlnf",
    version=__version__,
    description="RL from Human Feedback for ASR",
    author="RobotsMali AI4D Lab",
    author_email="diarray@robotsmali.org",
    license="MIT",
    packages=[
        "rlnf",
        "rlnf.dataloaders",
        "rlnf.ppo",
        "rlnf.reward",
    ],
    package_dir={
        "rlnf": ".",
        "rlnf.dataloaders": "dataloaders",
        "rlnf.ppo": "ppo",
        "rlnf.reward": "reward",
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)
