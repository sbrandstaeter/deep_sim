# -*- coding: utf-8 -*-
"""
Utility functions for the python2 to 3 wrapper.
"""


def object_to_id(obj):
    """
    Return list representing the cubit object. The first entry is the python id
    of the object, the second entry is the string representation.
    """
    return [
        'cp2t3id_' + str(id(obj)),
        str(obj)
        ]


def cubit_item_to_id(cubit_data_list):
    """Return the id from a cubit data list."""
    if not isinstance(cubit_data_list, list):
        return None
    if len(cubit_data_list) == 0:
        return None
    if not isinstance(cubit_data_list[0], str):
        return None
    if cubit_data_list[0].startswith('cp2t3id_'):
        return int(cubit_data_list[0][8:])
    else:
        return None


def is_base_type(obj):
    """
    Check if the object is of a base type that does not need conversion for the
    connection between python2 and python3.
    """
    if (isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float)
            or isinstance(obj, type(None))):
        return True
    else:
        return False
