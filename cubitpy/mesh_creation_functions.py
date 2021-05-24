# -*- coding: utf-8 -*-
"""
Implements functions that create basic meshes in cubit.
"""

# Python imports.
import numpy as np

# Import cubitpy stuff.
from . import cupy


def create_brick(cubit, h_x, h_y, h_z, *, element_type=None,
        mesh_interval=None, mesh_factor=None, mesh=True, **kwargs):
    """
    Create a cube in cubit.

    Args
    ----
    cubit: CubitPy
        CubitPy object.
    h_x, h_y, h_z: float
        size of the cube in x, y, and z direction.
    element_type: cubit.ElementType
        Type of the created element (HEX8, HEX20, HEX27, ...)
    mesh_interval: [int, int, int]
        Number of elements in each direction. This option is mutually
        exclusive with mesh_factor.
    mesh_interval: int
        Meshing factor in cubit. 10 is the largest. This option is mutually
        exclusive with mesh_factor.
    mesh: bool
        If the cube will be meshed or not.
    kwargs:
        Are passed to the Cubit.add_element_type function.
    """

    # Check if default value has to be set for element_type.
    if element_type is None:
        element_type = cupy.element_type.hex8

    # Check the input parameters.
    if h_x < 0 or h_y < 0 or h_z < 0:
        raise ValueError('Only positive lengths are possible!')
    if (mesh_interval is not None and mesh_factor is not None):
        raise ValueError('The keywords mesh_interval and mesh_factor are '
            + 'mutually exclusive!')

    # Create the block in cubit.
    solid = cubit.brick(h_x, h_y, h_z)
    volume_id = solid.volumes()[0].id()

    # Set the element type.
    cubit.add_element_type(solid.volumes()[0], el_type=element_type, **kwargs)

    # Set mesh properties.
    if mesh_interval is not None:

        # Get the lines in x, y and z direction.
        dir_curves = [[] for _i in range(3)]
        for curve in solid.curves():
            # Get the tangent on the line.
            tan = curve.tangent([0, 0, 0])
            for direction in range(3):
                # Project the tangent on the basis vector and check if it is
                # larger than 0.
                if np.abs(tan[direction]) > cupy.eps_pos:
                    dir_curves[direction].append(curve)
                    continue

        # Set the number of elements in x, y and z direction.
        for direction in range(3):
            string = ''
            for curve in dir_curves[direction]:
                string += ' {}'.format(curve.id())
            cubit.cmd('curve {} interval {} scheme equal'.format(string,
                mesh_interval[direction]))

    if mesh_factor is not None:
        # Set a cubit factor for the mesh size.
        cubit.cmd('volume {} size auto factor {}'.format(
            volume_id, mesh_factor))

    # Mesh the created block.
    if mesh:
        cubit.cmd('mesh volume {}'.format(volume_id))

    return solid
