# -*- coding: utf-8 -*-
"""
Utility functions for cubitpy.
"""

# Python imports.
import os


def check_environment_eclipse():
    """
    Adapt the environment path given to python by eclipse to call cubit.
    """

    # Flag if the script is run with eclipse or not. This will temporary
    # delete the python path so that the python2 interpreter does not look
    # in the wrong directories.
    # https://stackoverflow.com/questions/3248271/eclipse-using-multiple-python-interpreters-with-execnet
    # Also the console output will not be redirected to the eclipse console
    # but the path to a other console should be explicitly given if needed.

    if ('PYTHONPATH' in os.environ.keys()
            and 'pydev' in os.environ['PYTHONPATH']):
        python_path_old = os.environ['PYTHONPATH']
        python_path_new_list = []
        for item in python_path_old.split(':'):
            if (('/input' in item) or ('/cubitpy' in item)
                    or ('/meshpy' in item)):
                python_path_new_list.append(item)
        os.environ['PYTHONPATH'] = ':'.join(python_path_new_list)
        return True, python_path_old
    else:
        return False, ''
