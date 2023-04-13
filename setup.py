#setup file

from setuptools import setup, find_packages

setup(
    name = 'lensbiases',
    version = '0.0.1',
    description = 'LSS biases for CMB lensing.',
    url = 'https://github.com/Saladino93/lensbiases',
    packages = ['lensbiases'],
    package_dir = {'lensbiases': 'lensbiases'}
    )