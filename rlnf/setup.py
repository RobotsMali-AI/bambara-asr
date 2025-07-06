from setuptools import setup, find_packages
from rlnf import __version__  # Import version from rlnf package

with open("requirements.txt", encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='rlnf',
    version=__version__, # Update version as needed
    packages=find_packages(),
    description='An implementation of Reinforcement Learning from Human Feedback for ASR',
    install_requires=requirements,
    author='RobotsMali AI4D Lab',
    author_email='diarray@robotsmali.org',
    license='MIT',
)
