#!/usr/bin/env python

import os
import setuptools
from setuptools import find_packages, setup

# Utilty function to read the README file
def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="conform_agent",
    version="0.0.1",
    author="Florian Schulze",
    author_email="flori.schulze@yahoo.de",
    description="A package to support concept formation research using reinforcement "
                "learning. It provides Models, experimental seutups and environments.",
    url="https://github.com/JohnBergago/ConFormAgent",
    license="MIT",
    long_description=read('../README.md'),
    long_description_content_type='text/markdown',
    packages=["conform_agent"],
    #package_data="",
    install_requires=[
        "tensorflow<2.3",
        "torch",
        "ray[all]>=1.0,<=1.1",
        "mlagents==0.22.0",
        "mlagents-envs==0.22.0"
    ],
    python_requires='>=3.6',
)
