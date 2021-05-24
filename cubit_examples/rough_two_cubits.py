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


current_path = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(current_path, 'rough_block_output')

def create_output_dir():
    """Check if the temp directory exists, if not create it."""
    os.makedirs(file_path, exist_ok=True)
    return file_path

def rough_block(cubit, z_surf, nx, ny, nz):
    """
    Create the solid mesh for the plane curved test.

    Return groups with the nodes at the bottom and top surface.
    """

    def _get_node_index(i_x, i_y, i_z):
        return (nz + 1) * (ny + 1) * i_x + (nz + 1) * i_y + i_z

    def _get_element_index(i_x, i_y, i_z):
        return nz * ny * i_x + nz * i_y + i_z

    x_range = np.linspace(0.,1.0,nx+1)
    y_range = np.linspace(0.,1.0,ny+1) 
    # Create the nodes.
    coordinates = np.zeros([(nx + 1) * (ny + 1) * (nz + 1), 3])
    for i_x in range(nx+1):
        x = x_range[i_x]
        for i_y in range(ny+1):
            y = y_range[i_y]
            z_top = z_surf[i_x*(nx+1)+i_y] + 1.0
            for i_z in range(nz+1):
                z = i_z * z_top / nz
                if (z < 0):
                    raise ValueError('The function should be positive in the '
                        'whole block.')
                coordinates[_get_node_index(i_x, i_y, i_z), :] = [x, y, z[0]]

    # Create the elements.
    topology = np.zeros([nx * ny * nz, 8], dtype=int)
    for i_x in range(nx):
        for i_y in range(ny):
            for i_z in range(nz):
                element_index = _get_element_index(i_x, i_y, i_z)
                node_index_1 = _get_node_index(i_x, i_y, i_z)
                node_index_2 = _get_node_index(i_x + 1, i_y, i_z)
                node_index_3 = _get_node_index(i_x + 1, i_y + 1, i_z)
                node_index_4 = _get_node_index(i_x, i_y + 1, i_z)
                topology[element_index, :] = [
                    node_index_1,
                    node_index_2,
                    node_index_3,
                    node_index_4,
                    node_index_1 + 1,
                    node_index_2 + 1,
                    node_index_3 + 1,
                    node_index_4 + 1
                    ]

    # Get the start index for hex and nodes.
    i_node_base = cubit.get_node_count()
    i_element_base = cubit.get_hex_count()
    for coord in coordinates:
        coord_string = ' '.join(map(str, coord))
        cubit.cmd('create node location {}'.format(coord_string))
    for element_topology in topology:
        element_string = ' '.join(map(str, i_node_base + element_topology + 1))
        cubit.cmd('create hex node {}'.format(element_string))

    # Return the groups containing relevant geometry items.
    bottom = cubit.group()
    top = cubit.group()
    for i_x in range(nx + 1):
        for i_y in range(ny + 1):
            bottom_index = _get_node_index(i_x, i_y, 0)
            bottom.add('add node {}'.format(bottom_index + i_node_base + 1))
            top_index = _get_node_index(i_x, i_y, nz)
            top.add('add node {}'.format(top_index + i_node_base + 1))
    #all_nodes = cubit.group(name='function_block_all')
    #for i in range(len(coordinates)):
    #    all_nodes.add('add node {}'.format(i + i_node_base + 1))
    elements = cubit.group(name='Block_rough')
    for i in range(len(topology)):
        elements.add('add hex {}'.format(i + i_element_base + 1))

    cubit.add_element_type(elements, cupy.element_type.hex8,
            name=None, material='MAT 2',
            bc_description='KINEM nonlinear EAS none')

    # apply BC for the rough surface
    # dirichlet
    cubit.add_node_set(bottom, name='fix',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 3 ONOFF 1 1 1 '
                        + 'VAL 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0')
    
    cubit.add_node_set(top, name='slave',
                    bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                    bc_description='1 Slave')


    return {
        #'all': all_nodes,
        'top': top,
        'bottom': bottom,
        'hex': elements
        }



def generate_blocks():
    """
    """
    cubit = CubitPy()

    
    # upper flat block
    block_size = [1, 1, 1]
    n_elements = [1, 1, 1]

    block_flat = create_brick(cubit, block_size[0], block_size[1], block_size[2], mesh_interval=[n_elements[0], n_elements[1], n_elements[2]],
        mesh=False)

    movement = [0.5,0.5,1.7]
    cubit.move(block_flat, movement)

    block_flat.mesh()

    '''
    # upper block
    block_2_dim = [2, 2, 2]
    block_2 = create_brick(cubit, block_2_dim[0], block_2_dim[1], block_2_dim[2], mesh_interval=[1, 1, 1],
        mesh=False)
    movement = (block_2_dim[2] + block_1_dim[2]) / 2 + block_2_dim[2]*0.1
    cubit.move(block_2, [0.0, 0.0, movement])
    cubit.cmd("rotate Volume 2 about z angle 55.")

    # Mesh the blocks.
    block_1.mesh()

    # create the upper flat block
    
    block_size = [1, 1, 1]
    n_elements = [1, 1, 1]

    lx, ly, lz = block_size
    nx, ny, nz = n_elements

    block_flat = cubit.brick(lx, ly, lz)

    # Move the block.
    movement = [0.5,0.5,1.7]
    cubit.move(block_flat, movement)

    # Set the meshing parameters for the curves.
    for line in block_flat.curves():
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
    block_flat.mesh()
    cubit.add_element_type(block_flat.volumes()[0], cupy.element_type.hex8,
        name=None, material='MAT 2',
        bc_description='KINEM nonlinear EAS none')
    '''
    
    # define the BC for the flat block
    for _, surf in enumerate(block_flat.surfaces()):
        normal = np.array(surf.normal_at(get_surface_center(surf)))                
        if np.dot(normal, [0, 0, 1]) == 1:
            cubit.add_node_set(surf, name='load_upper',
                bc_section='DESIGN SURF DIRICH CONDITIONS',
                bc_description='NUMDOF 3 ONOFF 1 1 1 '
                    + 'VAL 0.0 0.0 -0.6 '
                    + 'FUNCT 1 1 2 ')
        elif np.dot(normal, [0, 0, 1]) == -1:
            cubit.add_node_set(surf, name='master',
                bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                bc_description='1 Master')

    # import the rough surface topology
    filename = "sup2.dat"
    size_power = 2**int(filename[3])
    current_dir = os.getcwd()
    filepath=os.path.join(current_dir,filename)


    z_surf = np.loadtxt(fname=filepath,delimiter=";",usecols=range(size_power+1))
    z_surf = z_surf.reshape((size_power+1)*(size_power+1),1)/3000.0

    # create the lower rough block
    rough_block(cubit, z_surf, size_power, size_power, size_power)

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
    generate_blocks()
