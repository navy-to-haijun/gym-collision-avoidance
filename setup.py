from setuptools import setup, find_packages
from io import open
import os


install_requires = [
        'tensorflow >=1.14, <2.0',
        'Pillow',
        'PyOpenGL',
        'pyyaml',
        'matplotlib',
        'pytz',
        'imageio==2.4.1',
        'gym',
        'moviepy',
        'pandas',
    ]

setup(
    name='gym_collision_avoidance',
    version='0.0.2',
    description='Simulation environment for collision avoidance',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mit-acl/gym-collision-avoidance',
    author='Michael Everett, Yu Fan Chen, Jonathan P. How, MIT',  # Optional
    keywords='robotics planning gym rl',  # Optional
    python_requires='<3.8',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
