#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'traitlets',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
    'numpy',
    'traitlets',
]

setup(
    name='homog',
    version='0.1.11',
    description="Utilities for Homogeneous Coordinates with Numpy",
    long_description=readme + '\n\n' + history,
    author="Will Sheffler",
    author_email='willsheffler@gmail.com',
    url='https://github.com/willsheffler/homog',
    packages=find_packages(include=['homog']),
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='homog',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
