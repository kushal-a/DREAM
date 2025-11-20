import io
import os
import re
from setuptools import setup, find_packages

setup(
    name='posediff',
    version='1.0.0',
    author='Kushal Agarwal',
    author_email='kushalagarwal444@gmail.com',
    description='Modifications to PoseDiff Model',
    packages=['posediff'],
    package_dir={'posediff': 'posediff'},
    zip_safe=False,
)