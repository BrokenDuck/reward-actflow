#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='active_pretraining',
      version='0.1',
      description='Active Exploration for Flow Models',
      author='misc',
      packages=find_packages(include=['active_pretraining', 'active_pretraining.*']),
      include_package_data=True
     )
