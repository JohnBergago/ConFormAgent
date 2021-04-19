#!/usr/bin/env python

import os
import setuptools
from setuptools import find_packages, setup

# Utilty function to read the README file
def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()

def package_files(directory):
    paths = []
    for (path, directories, filnames) in os.walk(directory):
        for filename in filnames:
            paths.append(os.path.join("..", path, filename))
    return paths

unity_executables = package_files('conform_agent/unity_executables')

setup(
    name="conform_agent",
    version="0.0.3",
    author="Florian Schulze",
    author_email="flori.schulze@yahoo.de",
    description="A package to support concept formation research using reinforcement "
                "learning. It provides Models, experimental seutups and environments.",
    url="https://github.com/JohnBergago/ConFormAgent",
    license="MIT",
    long_description=read('../README.md'),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "": ["LICENSE", "README.md"],
        "unity_executables": unity_executables,
    },
    install_requires=[
        "tensorflow>=2.0",
        "torch",
        "ray[all]>=1.2.*",
        "mlagents==0.24.1",
        "mlagents-envs==0.24.01"
    ],
    python_requires='>=3.6',

    entry_points={
        "console_scripts": [
            "conform-rllib=conform_agent.scripts:cli",
        ]
    },
)
