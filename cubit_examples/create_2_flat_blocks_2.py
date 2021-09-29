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

        # the idea
        '''
 
        * * * * * * * * * * * * * 
        *       *       *       *
        *       *       *       *
        *       *       *       *
        * * * * * * * * * * * * *
        *       *       *       *
        *       *       *       *   --> block 1
        *       *       *       *
        * * * * * * * * * * * * * 
        *       *       *       *
        *       *       *       *
        *       *       *       *
        * * * * * * * * * * * * *


        * * * * * * * * * * * * *   -----------
        *   *   *   *   *   *   *
        * * * * * * * * * * * * *   --> block 2
        *   *   *   *   *   *   *
        * * * * * * * * * * * * *   ------------
        *       *       *       *
        *       *       *       *
        *       *       *       *
        * * * * * * * * * * * * *   --> block 3
        *       *       *       *
        *       *       *       *
        *       *       *       *
        * * * * * * * * * * * * *


        '''
        # initialize the cubit
        cubit = CubitPy()

        # ----------------------------------------------------------
        # create the first block
        block_1_size = [100, 100, 80]
        block_1_n_elements = [5, 5, 5] # the number of elements on each edge

        lx_1, ly_1, lz_1 = block_1_size
        nx_1, ny_1, nz_1 = block_1_n_elements
    
        block_1 = cubit.brick(lx_1, ly_1, lz_1) # creates the brick element

        # Set the meshing parameters for the curves 
        # seed for the mesh
        for line in block_1.curves():
            point_on_line = line.position_from_fraction(0.5)
            tangent = np.array(line.tangent(point_on_line))
            if np.abs(np.dot(tangent, [1, 0, 0])) > 1e-5:
                cubit.set_line_interval(line, nx_1)
            elif np.abs(np.dot(tangent, [0, 1, 0])) > 1e-5:
                cubit.set_line_interval(line, ny_1)
            elif np.abs(np.dot(tangent, [0, 0, 1])) > 1e-5:
                cubit.set_line_interval(line, nz_1)
            else:
                raise ArithmeticError('Error')
        
        # mesh the block
        block_1.mesh()
        cubit.add_element_type(block_1.volumes()[0], cupy.element_type.hex8,
            name='block_1', material='MAT 1',
            bc_description='KINEM nonlinear EAS none')
        
        # -------------------------------------------------------------
        # repeat the same procedure above
        block_2_size = [100, 100, 20]
        block_2_n_elements = [5, 5, 5]

        lx_2, ly_2, lz_2 = block_2_size
        nx_2, ny_2, nz_2 = block_2_n_elements
    
        block_2 = cubit.brick(lx_2, ly_2, lz_2)

        movement = (lz_2 + lz_1) / 2 # move the middle block 
        # how is the movement done?
        # try to sketch that whenever you create a block, the center of the block is created at the origin. So you have to move the block half of the sum of the z lengths
        cubit.move(block_2, [0.0, 0.0, movement])

        # Set the meshing parameters for the curves.
        for line in block_2.curves():
            point_on_line = line.position_from_fraction(0.5)
            tangent = np.array(line.tangent(point_on_line))
            if np.abs(np.dot(tangent, [1, 0, 0])) > 1e-5:
                cubit.set_line_interval(line, nx_2)
            elif np.abs(np.dot(tangent, [0, 1, 0])) > 1e-5:
                cubit.set_line_interval(line, ny_2)
            elif np.abs(np.dot(tangent, [0, 0, 1])) > 1e-5:
                cubit.set_line_interval(line, nz_2)
            else:
                raise ArithmeticError('Error')
        
        block_2.mesh()
        cubit.add_element_type(block_2.volumes()[0], cupy.element_type.hex8,
            name='block_2', material='MAT 2',
            bc_description='KINEM nonlinear EAS none')

        #---------------------------------------------------------------------
        # repeat the same procedure above
        block_3_size = [100, 100, 100]
        block_3_n_elements = [5, 5, 5]

        lx_3, ly_3, lz_3 = block_3_size
        nx_3, ny_3, nz_3 = block_3_n_elements
    
        block_3 = cubit.brick(lx_3, ly_3, lz_3)

        movement = (lz_3 + lz_1) / 2 + lz_2 + lz_2*0.05  # the last term gives the gap
        cubit.move(block_3, [0.0, 0.0, movement])

        # Set the meshing parameters for the curves.
        for line in block_3.curves():
            point_on_line = line.position_from_fraction(0.5)
            tangent = np.array(line.tangent(point_on_line))
            if np.abs(np.dot(tangent, [1, 0, 0])) > 1e-5:
                cubit.set_line_interval(line, nx_3)
            elif np.abs(np.dot(tangent, [0, 1, 0])) > 1e-5:
                cubit.set_line_interval(line, ny_3)
            elif np.abs(np.dot(tangent, [0, 0, 1])) > 1e-5:
                cubit.set_line_interval(line, nz_3)
            else:
                raise ArithmeticError('Error')
        
        block_3.mesh()
        cubit.add_element_type(block_3.volumes()[0], cupy.element_type.hex8,
            name='block_3', material='MAT 3',
            bc_description='KINEM nonlinear EAS none')


        # Create node sets.
        # block 1 should have fixed bc and tying contact node set (master_coupling)
        for _, surf in enumerate(block_1.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))
            if np.dot(normal, [0, 0, 1]) == -1:
                cubit.add_node_set(surf, name='fix_bottom',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 3 ONOFF 1 1 1 '
                        + 'VAL 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0')
            elif np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='master_coupling',
                    bc_section='DESIGN SURF MORTAR COUPLING CONDITIONS 3D',
                    bc_description='1 Master Inactive')
        # block 2 should have tying contact node set (slave_coupling) and the slave contact (high mesh means slave)
        for _, surf in enumerate(block_2.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))                
            if np.dot(normal, [0, 0, 1]) == -1:
                cubit.add_node_set(surf, name='slave_coupling',
                    bc_section='DESIGN SURF MORTAR COUPLING CONDITIONS 3D',
                    bc_description='1 Slave Active')
            elif np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='slave_contact',
                    bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                    bc_description='1 Slave Inactive')
        # block 3 should have the applied disp in y and z direction and the master contact (low mesh means master)
        for _, surf in enumerate(block_3.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))                
            if np.dot(normal, [0, 0, 1]) == -1:
                cubit.add_node_set(surf, name='master_contact',
                    bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                    bc_description='1 Master Inactive')
            elif np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='disp_up',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 3 ONOFF 1 1 1 '
                        + 'VAL 0.0 1.0 -1.0 '
                        + 'FUNCT 0 1 1')
                            

        
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
   -------------------------------------------------------PROBLEM TYP
    PROBLEMTYP   Structure
    RESTART      0
    ----------------------------------------------------DISCRETISATION
    NUMFLUIDDIS  0
    NUMSTRUCDIS  1
    NUMALEDIS    0
    NUMTHERMDIS  0
    ----------------------------------------------------------------IO
    OUTPUT_BIN        Yes
    STRUCT_DISP       Yes
    STRUCT_STRESS     Cauchy
    FLUID_SOL         No
    FLUID_STRESS      No
    FLUID_VIS         No
    FILESTEPS         1000
    ------------------------------------------------STRUCTURAL DYNAMIC
    LINEAR_SOLVER         1
    DYNAMICTYP            Statics
    RESULTSEVRY           1
    RESTARTEVRY           1
    NLNSOL                fullnewton
    TIMESTEP              0.01
    NUMSTEP               20
    MAXTIME               1.0
    TOLRES                1.0E-5
    TOLDISP               1.0E-8
    NORM_RESF             Abs
    NORM_DISP             Abs
    NORMCOMBI_RESFDISP    And
    MAXITER               50
    PREDICT               ConstVel
    INT_STRATEGY          Standard
    -------------------------------------------------------------CONTACT DYNAMIC
    LINEAR_SOLVER        2
    SYSTEM               condensed
    SEMI_SMOOTH_NEWTON   Yes
    SEMI_SMOOTH_CN       10.0
    STRATEGY             Lagrange
    FRICTION                        None
    -------------------------------------------------------------MORTAR COUPLING
    SEARCH_ALGORITHM     Binarytree
    SEARCH_PARAM         0.5
    LM_SHAPEFCN          pg
    -------------------------------------MORTAR COUPLING/PARALLEL REDISTRIBUTION
    PARALLEL_REDIST      dynamic
    ----------------------------------------------------------SOLVER 1
    NAME                 Structure_Solver
    SOLVER               Superlu//UMFPACK
    ----------------------------------------------------------SOLVER 2
    NAME                 Contact_Solver
    SOLVER               Superlu//UMFPACK
    -------------------------------------------------------------------MATERIALS
    MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 500.0 NUE 0.1 DENS 0.02 THEXPANS 0.0
    MAT 2 MAT_Struct_StVenantKirchhoff YOUNG 500.0 NUE 0.3 DENS 0.02 THEXPANS 0.0
    MAT 3 MAT_Struct_StVenantKirchhoff YOUNG 5000.0 NUE 0.3 DENS 0.02 THEXPANS 0.0
    ----------------------------------------------------------------------FUNCT1
    COMPONENT 0 FUNCTION a
    VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 TIMES 0 0.5 1 VALUES 0 30 30
    ----------------------------------------------------------------------FUNCT2
    COMPONENT 0 FUNCTION a
    VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 TIMES 0 0.5 1 VALUES 0 0 40
    '''

    return header


if __name__ == '__main__':
    """Execution part of script."""

    #create_patch_test(False)
    two_cubits_contact()
