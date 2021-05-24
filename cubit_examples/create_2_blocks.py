# -*- coding: utf-8 -*-
"""
This script generates patch tests for beam to solid surface mesh tying.
"""

# Python modules.
import numpy as np
import os
import fileinput
import shutil
import subprocess

# Import cubitpy module.
from cubitpy import CubitPy, cupy, get_surface_center
from cubitpy.mesh_creation_functions import create_brick

current_path = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(current_path, 'two_block_output')

def create_output_dir():
    """Check if the temp directory exists, if not create it."""
    os.makedirs(file_path, exist_ok=True)
    return file_path


def two_cubits_contact():
        """Create node-node and vertex-vertex coupling."""

        # First create two blocks.
        # upper block
        cubit = CubitPy()
        block_1 = create_brick(cubit, 1, 1, 1, mesh_interval=[2, 2, 2],
            mesh=False)
        
        # lower-block
        block_2 = create_brick(cubit, 1, 1, 1, mesh_interval=[2, 2, 2],
            mesh=False)
        cubit.move(block_2, [0.0, 0.0, 1.2])

        # Mesh the blocks.
        block_1.mesh()
        block_2.mesh()

        # Create node sets.
        for i, surf in enumerate(block_1.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))
            if np.dot(normal, [0, 0, -1]) == 1:
                cubit.add_node_set(surf, name='fix',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 3 ONOFF 1 1 1 '
                        + 'VAL 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0')
            elif np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='master',
                    bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                    bc_description='1 Master')
        
        for i, surf in enumerate(block_2.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))                
            if np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='load_upper',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 3 ONOFF 1 1 1 '
                        + 'VAL 0.0 0.0 -0.3 '
                        + 'FUNCT 1 1 2 ')
            elif np.dot(normal, [0, 0, 1]) == -1:
                cubit.add_node_set(surf, name='slave',
                    bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                    bc_description='1 Slave')

        output_dir = create_output_dir()
        # create dat, exo and cub files
        file_name = 'two_cubits_contact'
        final_path_names = os.path.join(output_dir, file_name)
        
        cubit.export_cub(final_path_names + '.cub')
        cubit.export_exo(final_path_names + '.exo')


        with open(final_path_names+".bc", 'w') as bc_file:
            bc_file.write('---------------------------------------BCSPECS\n\n')
            for i, block in enumerate(cubit.blocks):
                bc_file.write((
                    '*eb{}="ELEMENT"\nsectionname="{}"\n'
                    + 'description="{}"\nelementname="{}"\n\n').format(
                        i + 1, block[0], block[1], block[2]))
            for i, node_set in enumerate(cubit.node_sets):
                bc_file.write((
                    '*ns{}="CONDITION"\nsectionname="{}"\n'
                    + 'description="{}"\n\n').format(
                        i + 1, node_set[0], node_set[1]))
        
        contact_header = os.path.join(current_path,"contact_header")
        re_name = os.path.join(file_path,(file_name+".head"))
        shutil.copy(contact_header,re_name)

        out = subprocess.check_output([
            cubit.pre_exodus,
            '--exo='+file_name+'.exo',
            '--bc='+file_name+'.bc',
            '--head='+file_name+'.head'
            ], cwd=output_dir)
            

    #cubit.create_dat(os.path.join(file_path, file_name) + '.dat')
        #cubit.display_in_cubit()


if __name__ == '__main__':
    """Execution part of script."""

    #create_patch_test(False)
    two_cubits_contact()
