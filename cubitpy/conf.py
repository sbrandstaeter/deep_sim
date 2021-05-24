# -*- coding: utf-8 -*-
"""
This module defines a global object that manages all kind of stuff regarding
cubitpy.
"""

# Python imports.
import os
import getpass

# Cubitpy imports.
from .cubitpy_types import (FiniteElementObject, GeometryType, ElementType,
    CubitItems, BoundaryConditionType)


class CubitOptions(object):
    """Object for types in cubitpy."""
    def __init__(self):

        # Temporary directory for cubitpy.
        self.temp_dir = os.path.join('/tmp/cubitpy', '{}_{}'.format(
            getpass.getuser(), os.getpid()))
        self.temp_log = os.path.join(self.temp_dir, 'cubitpy.log')

        # Check if temp path exits, if not create it.
        os.makedirs(self.temp_dir, exist_ok=True)

        # Geometry types.
        self.geometry = GeometryType

        # Finite element types.
        self.finite_element_object = FiniteElementObject

        # Element shape types.
        self.element_type = ElementType

        # Cubit internal items.
        self.cubit_items = CubitItems

        # Boundary condition type.
        self.bc_type = BoundaryConditionType

        # Tolerance for geometry.
        self.eps_pos = 1e-10

    @staticmethod
    def get_default_paths(name, throw_error=True):
        """Look for and return a path to cubit or pre_exodus."""

        if name == 'cubit':
            environment_variable = 'CUBIT'
            test_function = os.path.isdir
        elif name == 'pre_exodus':
            environment_variable = 'BACI_PRE_EXODUS'
            test_function = os.path.isfile
        else:
            raise ValueError('Type {} not implemented!'.format(name))

        # Check if he environment variable is set and the path exits.
        if environment_variable in os.environ.keys():
            if test_function(os.environ[environment_variable]):
                return os.environ[environment_variable]

        # No valid path found or given.
        if throw_error:
            raise ValueError('Path for {} not found!'.format(name))
        else:
            return None


# Global object with options for cubitpy.
cupy = CubitOptions()
