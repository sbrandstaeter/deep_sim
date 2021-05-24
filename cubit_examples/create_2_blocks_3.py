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
        # lower block
        cubit = CubitPy()
        block_1_dim = [10 ,10 ,2]
        block_1 = create_brick(cubit, block_1_dim[0], block_1_dim[1], block_1_dim[2], mesh_interval=[block_1_dim[0], block_1_dim[1], block_1_dim[2]],
            mesh=False)
        
        # upper block
        block_2_dim = [2, 2, 2]
        block_2 = create_brick(cubit, block_2_dim[0], block_2_dim[1], block_2_dim[2], mesh_interval=[1, 1, 1],
            mesh=False)
        movement = (block_2_dim[2] + block_1_dim[2]) / 2 + block_2_dim[2]*0.1
        cubit.move(block_2, [0.0, 0.0, movement])
        cubit.cmd("rotate Volume 2 about z angle 55.")

        # Mesh the blocks.
        block_1.mesh()
        block_2.mesh()


        # Create node sets.
        for _, surf in enumerate(block_1.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))
            if np.dot(normal, [0, 0, -1]) == 1:
                continue
                cubit.add_node_set(surf, name='fix',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 3 ONOFF 1 1 1 '
                        + 'VAL 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0')
            elif np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='master',
                    bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                    bc_description='1 Master')
        
        for _, surf in enumerate(block_2.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))                
            if np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='load_upper',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 3 ONOFF 1 1 1 '
                        + 'VAL 0.0 0.0 -0.6 '
                        + 'FUNCT 1 1 2 ')
            elif np.dot(normal, [0, 0, 1]) == -1:
                cubit.add_node_set(surf, name='slave',
                    bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                    bc_description='1 Slave')
        '''
        cubit.add_node_set(
            solid.curves()[0],
            name='curve',
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 2')
            '''

        for _, surf in enumerate(block_1.surfaces()):
            surf_node_list = surf.get_node_ids()
            normal = np.array(surf.normal_at(get_surface_center(surf)))
            if np.dot(normal, [0, 0, -1]) == 1:
                for j, curv in enumerate(block_1.curves()):
                    curve_node_list = curv.get_node_ids()
                    if set(curve_node_list).issubset(surf_node_list):
                        cubit.add_node_set(
                        curv,
                        name='curve_{}'.format(j),
                        bc_section='DESIGN LINE DIRICH CONDITIONS',
                        bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0')
                            


        output_dir = create_output_dir()
        # create dat, exo and cub files
        file_name = 'two_cubits_contact'
        final_path_names = os.path.join(output_dir, file_name)
        
        cubit.export_cub(final_path_names + '.cub')
        cubit.export_exo(final_path_names + '.exo')

        cubit.head = get_contact_header()

        cubit.create_dat(final_path_names + '.dat')
        cubit.display_in_cubit()

def get_contact_header():

    header = '''
    -----------------------------------------------------------------PROBLEM TYP
    PROBLEMTYP                      Structure
    RESTART                         0
    --------------------------------------------------------------DISCRETISATION
    NUMFLUIDDIS                     0
    NUMSTRUCDIS                     1
    NUMALEDIS                       0
    NUMTHERMDIS                     0
    --------------------------------------------------------------------------IO
    OUTPUT_BIN                      Yes
    STRUCT_DISP                     Yes
    STRUCT_STRESS                   No
    FILESTEPS                       1000
    ----------------------------------------------------------STRUCTURAL DYNAMIC
    LINEAR_SOLVER                   1
    DYNAMICTYP                      GenAlpha
    RESULTSEVRY                     1
    RESTARTEVRY                     1
    NLNSOL                          fullnewton
    TIMESTEP                        0.02
    NUMSTEP                         30
    MAXTIME                         0.6
    DAMPING                         Yes
    M_DAMP                          0.00001
    K_DAMP                          0.00001
    TOLRES                          1.0E-8
    TOLDISP                         1.0E-8
    NORM_RESF                       Abs
    NORM_DISP                       Abs
    NORMCOMBI_RESFDISP              And
    MAXITER                         50
    INT_STRATEGY                    Standard
    -------------------------------------------------STRUCTURAL DYNAMIC/GENALPHA
    GENAVG                          TrLike
    RHO_INF                         1.0
    -------------------------------------------------------------CONTACT DYNAMIC
    LINEAR_SOLVER                   2
    STRATEGY                        Penalty
    PENALTYPARAM                    20000
    FRICTION                        None
    SEMI_SMOOTH_NEWTON              Yes
    SEMI_SMOOTH_CN                  1.0
    -------------------------------------------------------------MORTAR COUPLING
    LM_SHAPEFCN                     Standard
    SEARCH_ALGORITHM                BruteForceEleBased
    SEARCH_PARAM                    0.3
    LM_DUAL_CONSISTENT              none
    --------------------------------------------------------------------SOLVER 1
    NAME                            Structure_Solver
    SOLVER                          UMFPACK
    --------------------------------------------------------------------SOLVER 2
    NAME                            Contact_Solver
    SOLVER                          UMFPACK
    -------------------------------------------------------------------MATERIALS
    //                              MAT_Struct_StVenantKirchhoff
    MAT 1 MAT_ElastHyper NUMMAT 1 MATIDS 3 DENS 7.8e-6
    MAT 3 ELAST_CoupNeoHooke YOUNG 210.0 NUE 0.3
    MAT 2 MAT_ElastHyper NUMMAT 1 MATIDS 4 DENS 7.8e-6
    MAT 4 ELAST_CoupNeoHooke YOUNG 2100.0 NUE 0.3
    ----------------------------------------------------------------------FUNCT1
    COMPONENT 0 FUNCTION a
    VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 4 TIMES 0 0.5 0.6 1 VALUES 0 0 0.2 0.2
    ----------------------------------------------------------------------FUNCT2
    COMPONENT 0 FUNCTION a
    VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 TIMES 0 0.5 1 VALUES 0 1 1
    '''

    return header


if __name__ == '__main__':
    """Execution part of script."""

    #create_patch_test(False)
    two_cubits_contact()
