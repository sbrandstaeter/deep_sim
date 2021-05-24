# -*- coding: utf-8 -*-
"""
This script generates patch tests for beam to solid surface mesh tying.
"""

# Python modules.
import numpy as np
import os

# Import cubitpy module.
from cubitpy import CubitPy, cupy


def create_solid_plane(cubit):
    """
    Create the solid mesh for the plane patch test.

    Return groups with the bottom and top surface.
    """

    # Parameters.
    a = 1.0
    h = 1.0

    cubit.cmd('brick x {0} y {0} z {1}'.format(a, h))
    cubit.cmd('move volume 1 x 0 y 0 z {}'.format(0.5 * h))
    cubit.cmd(('webcut volume 1 with general plane location 0 0 {0} '
        + 'direction {1[0]} {1[1]} {1[2]} noimprint nomerge').format(
            0.5 * h, [0.3, 0.15, 1]))
    cubit.cmd('delete volume 1')
    cubit.cmd('volume 2 size auto factor 7')
    cubit.cmd('mesh volume 2')

    a = cubit.group()
    a.add('add volume 2')
    print(cubit.get_group_volumes(a._id))
    print(cubit.get_group_surfaces(a._id))


    bottom = cubit.group()
    bottom.add(cubit.surface(2))
    top = cubit.group()
    top.add(cubit.surface(12))
    return bottom, top


def create_brick_curved_top(cubit, b, h, z_function, nx, ny, nz):
    """
    Create the solid mesh for the plane curved test.

    Return groups with the nodes at the bottom and top surface.
    """

    def _get_node_index(i_x, i_y, i_z):
        return (nz + 1) * (ny + 1) * i_x + (nz + 1) * i_y + i_z

    def _get_element_index(i_x, i_y, i_z):
        return nz * ny * i_x + nz * i_y + i_z

    # Create the nodes.
    coordinates = np.zeros([(nx + 1) * (ny + 1) * (nz + 1), 3])
    for i_x in range(nx + 1):
        x = -0.5 * b + i_x * b / nx
        for i_y in range(ny + 1):
            y = -0.5 * h + i_y * h / ny
            z_top = z_function(x, y)
            for i_z in range(nz + 1):
                z = i_z * z_top / nz
                if (z < 0):
                    raise ValueError('The function should be positive in the '
                        'whole block.')
                coordinates[_get_node_index(i_x, i_y, i_z), :] = [x, y, z]

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
    bottom = cubit.group(name='function_block_bottom')
    top = cubit.group(name='function_block_top')
    for i_x in range(nx + 1):
        for i_y in range(ny + 1):
            bottom_index = _get_node_index(i_x, i_y, 0)
            bottom.add('add node {}'.format(bottom_index + i_node_base + 1))
            top_index = _get_node_index(i_x, i_y, nz)
            top.add('add node {}'.format(top_index + i_node_base + 1))
    all_nodes = cubit.group(name='function_block_all')
    for i in range(len(coordinates)):
        all_nodes.add('add node {}'.format(i + i_node_base + 1))
    elements = cubit.group(name='function_block_hex')
    for i in range(len(topology)):
        elements.add('add hex {}'.format(i + i_element_base + 1))
    return {
        'all': all_nodes,
        'top': top,
        'bottom': bottom,
        'hex': elements
        }


def create_solid_curved(cubit):
    """
    """

    # Parameters.
    a = 1.0
    h = 1.0

    def fun(x, y):
        """Function for the top face."""
        return h - x**2 - y**2

    return create_brick_curved_top(cubit, a, a, fun, 16, 16, 16)


def create_patch_test(curved):
    """
    """

    cubit = CubitPy()


    #cubit.cmd('brick x 0.8')
    #cubit.cmd('mesh volume 1')


    if curved:
        create_solid_curved(cubit)
    else:
        create_solid_plane(cubit)

   


    cubit.display_in_cubit()


if __name__ == '__main__':
    """Execution part of script."""

    #create_patch_test(False)
    create_patch_test(True)
