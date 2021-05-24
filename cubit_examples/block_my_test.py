# -*- coding: utf-8 -*-
"""
This script generates patch tests for beam to solid surface mesh tying.
"""

# Python modules.
import numpy as np
import os

# Cubitpy imports.
from cubitpy import CubitPy, cupy, get_surface_center
from cubitpy.mesh_creation_functions import create_brick


def create_block(cubit, dat_lines_compare=False, np_arrays=False):
        """
        Create a block with cubit.

        Args
        ----
        dat_lines_compare: bool
            If the created dat file should be compared or the list of lines
            returned by get_dat_lines.
        np_arrays: bool
            If the cubit interaction is with numpy or python arrays.
        """

        # Set head
        cubit.head = '''
            // Header processed by cubit.
            '''

        # Dimensions and mesh size of the block.
        block_size = [0.1, 1, 10]
        n_elements = [2, 4, 8]
        if np_arrays:
            lx, ly, lz = np.array(block_size)
            nx, ny, nz = np.array(n_elements)
        else:
            lx, ly, lz = block_size
            nx, ny, nz = n_elements

        # Create the block.
        block = cubit.brick(lx, ly, lz)

        # Move the block.
        move_array = [0, 0, block.bounding_box()[2]]
        if np_arrays:
            move_array = np.array(move_array)
        cubit.move(block, move_array)

        # Set the meshing parameters for the curves.
        for line in block.curves():
            point_on_line = line.position_from_fraction(0.5)
            tangent = np.array(line.tangent(point_on_line))
            if np.abs(np.dot(tangent, [1, 0, 0])) > 1e-5:
                cubit.set_line_interval(line, nx)
            elif np.abs(np.dot(tangent, [0, 1, 0])) > 1e-5:
                cubit.set_line_interval(line, ny)
            elif np.abs(np.dot(tangent, [0, 0, 1])) > 1e-5:
                cubit.set_line_interval(line, nz)
            else:
                raise ArithmeticError('Error')

        # Mesh the block.
        block.mesh()
        cubit.add_element_type(block.volumes()[0], cupy.element_type.hex8,
            name='block', material='MAT 1',
            bc_description='KINEM nonlinear EAS none')

        # Create node sets.
        for i, surf in enumerate(block.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))
            if np.dot(normal, [0, 0, -1]) == 1:
                cubit.add_node_set(surf, name='fix',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 6 ONOFF 1 1 1 0 0 0 '
                        + 'VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0 0 0 0')
            elif np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='load',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 6 ONOFF 1 1 1 0 0 0 '
                        + 'VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0 0 0 0')
            else:
                cubit.add_node_set(surf, name='load{}'.format(i),
                    bc_section='DESIGN SURF NEUMANN CONDITIONS',
                    bc_description='NUMDOF 6 ONOFF 1 1 1 0 0 0 '
                        + 'VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0 0 0 0')
        
        print(os.getcwd())
        filename = "/home/a11btasa/git_repos/cubitpy/temp_out/res.cub"
        filename_dat = "/home/a11btasa/git_repos/imcsml/cubit_trials/res.dat"
        #cubit.export_cub(filename)
        cubit.create_dat(filename_dat)
        #cubit.display_in_cubit()
        

def test_create_block():
    """
    Test the creation of a cubit block.
    """

    # Initialize cubit.
    cubit = CubitPy()
    create_block(cubit)

if __name__ == '__main__':
    """Execution part of script."""

    #create_patch_test(False)
    test_create_block()
    