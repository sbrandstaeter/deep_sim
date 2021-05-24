# -*- coding: utf-8 -*-
"""
Utility functions for the use of cubitpy.
"""


# Import cubitpy stuff.
from . import cupy


def get_surface_center(surf):
    """
    Get a 3D point that has the local coordinated on the surface of (0,0),
    with the parameter space being ([-1,1],[-1,1]).
    """

    if not surf.get_geometry_type() == cupy.geometry.surface:
        raise TypeError('Did not expect {}'.format(type(surf)))

    range_u = surf.get_param_range_U()
    u = 0.5 * (range_u[1] + range_u[0])
    range_v = surf.get_param_range_V()
    v = 0.5 * (range_v[1] + range_v[0])
    return surf.position_from_u_v(u, v)
