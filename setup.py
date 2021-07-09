import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="deep_sim",
    version="0.1",
    author="Tarik Sahin",
    author_email="tarik.sahin@unibw.de",
    description=("Machine Learning Automatization Tool"),
    keywords="ML, DL",
    packages=[
        'pydeep_sim',
        'pydeep_sim.iterators',
        'pydeep_sim.rough_surface',
        'pydeep_sim.sampling'
    ],

    long_description=read('README.md'),
    tests_require='pytest',
)
