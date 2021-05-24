# -*- coding: utf-8 -*-
"""
This script generates patch tests for beam to solid surface mesh tying.
"""

# Python modules.
import numpy as np
import os
import fileinput

# Import cubitpy module.
from cubitpy import CubitPy, cupy, get_surface_center
from cubitpy.mesh_creation_functions import create_brick

testing_path = os.path.abspath(os.path.dirname(__file__))
testing_temp = os.path.join(testing_path, 'testing-tmp')

def check_tmp_dir():
    """Check if the temp directory exists, if not create it."""
    os.makedirs(testing_temp, exist_ok=True)


def compare(cubit, name, dat_lines_compare=False,
            single_precision=False):
        """
        Write create the dat file from the cubit mesh and compare to a
        reference file.

        Args
        ----
        cubit: Cubit object.
        name: str
            Name of the test case. A reference file 'name' + '_ref.dat' must
            exits in the reference file folder.
        dat_lines_compare: bool
            If the created file should be compared or the list of lines
            returned by get_dat_lines.
        single_precision: bool
            If the output of cubit is single or double precision.
        """

        # Create the dat file for the solid.
        check_tmp_dir()

        if single_precision:
            cubit.cmd('set exodus single precision on')

        # Get the string of the input file, depending on the chosen method.
        if not dat_lines_compare:
            dat_file = os.path.join(testing_temp, name + '.dat')
            cubit.create_dat(dat_file)
            with open(dat_file, 'r') as text_file:
                string2 = text_file.read()
        else:
            string2 = ''.join(cubit.get_dat_lines())

        # Compare with the ref file.
        #ref_file = os.path.join(testing_input, name + '_ref.dat')
        #with open(ref_file, 'r') as text_file:
        #    string1 = text_file.read()
        #self.assertTrue(
        #    compare_strings(string1, string2), name)

def node_set_geometry_type():
        """Create the boundary conditions via the bc_type enum."""

        # First create the solid mesh.
        cubit = CubitPy()
        solid = create_brick(cubit, 1, 1, 1, mesh_interval=[1, 1, 1])

        # Add all possible boundary conditions.

        # Dirichlet and Neumann.
        cubit.add_node_set(
            solid.vertices()[0],
            name='vertex',
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 1')
        cubit.add_node_set(
            solid.curves()[0],
            name='curve',
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 2')
        cubit.add_node_set(
            solid.surfaces()[0],
            name='surface',
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 3')
        cubit.add_node_set(
            solid.volumes()[0],
            name='volume',
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 4')

        # Define boundary conditions on explicit nodes.
        cubit.add_node_set(
            cubit.group(add_value='add node 2'),
            name='point2',
            geometry_type=cupy.geometry.vertex,
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 4')
        cubit.add_node_set(
            cubit.group(add_value='add node {}'.format(' '.join(
                [str(i + 1) for i in range(cubit.get_node_count())]))),
            name='point3',
            geometry_type=cupy.geometry.vertex,
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 4')

        '''
        # Coupling.
        cubit.add_node_set(
            solid.volumes()[0],
            name='coupling_btsv',
            bc_type=cupy.bc_type.beam_to_solid_volume_meshtying,
            bc_description='COUPLING_ID 1'
            )
        cubit.add_node_set(
            solid.surfaces()[0],
            name='coupling_btss',
            bc_type=cupy.bc_type.beam_to_solid_surface_meshtying,
            bc_description='COUPLING_ID 1'
            )
        '''
        # Set the head string.
        cubit.head = '''
            ----------------------------------------------------------MATERIALS
            MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 10 NUE 0.0 DENS 0.0'''

        # Compare the input file created for baci.
        compare(cubit, 'test_node_set_geometry_type')


if __name__ == '__main__':
    """Execution part of script."""

    #create_patch_test(False)
    node_set_geometry_type()