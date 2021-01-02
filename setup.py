#!/usr/bin/env python

import setuptools
from setuptools import find_packages, setup

setup(
    name="conform",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)