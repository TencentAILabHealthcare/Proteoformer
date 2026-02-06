# setup.py
from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    """Parse requirements from requirements.txt file"""
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as file:
        lines = file.readlines()
        reqs = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return reqs

setup(
    name='proteoformer',
    version='1.0',
    description='A proteoform language model for proteoform sequence analysis',
    author='devinjzhu',
    # author_email='',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    # install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)