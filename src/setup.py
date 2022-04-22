#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='otscomics',
      version='0.1.0',
      description='Distances between cells for single-cell omics with optimal transport',
      author='Geert-Jan Huizing',
      author_email='huizing@ens.fr',
      packages=['otscomics'],
      install_requires=[
        'pot>=0.8',
        'torch>=1.0',
        'numpy>=1.20',
        'scipy>=1.6',
        'tqdm>=4.62'
      ]
    )
