# -*- coding: utf-8 -*-
"""
This script generates patch tests for beam to solid surface mesh tying.
"""

# Python modules.
import numpy as np
import os
import fileinput
import numpy as np
import sys

# Import cubitpy module.
from cubitpy import CubitPy, cupy



# https://github.com/geodynamics/specfem3d/blob/master/EXAMPLES/Mount_StHelens/mesh_mount_stl.py

#!python



def rough_surface(filename):

    cubit = CubitPy()

    count = 0
    y_pos = 0
    for line in fileinput.input(filename):
        if count == 0:
            lineitems = line.split(";")
            size = len(lineitems)-1
            x_mesh = np.linspace(0,1000,size)
            y_mesh = np.linspace(0,1000,size)

        lineitems = line.split(";")
        x_pos = 0
        lineitems.pop()
        
        for each_item in lineitems:
            x = x_mesh[x_pos]
            y = y_mesh[y_pos]
            z = each_item.strip()
            xyz = str(x) + ' ' + str(y) + ' ' + str(z)
            count += 1
            x_pos += 1 
            cubit.cmd('create vertex '+ xyz )

        y_pos += 1


    fileinput.close()


    nstep = 33
    # creates smooth spline curves for surface
    
    countcurves = 0
 
    for y_pos in range(size): #for the y_position
        for x_pos in range(size-1):
            if y_pos != size-1:
                cubit.cmd('create curve vertex ' + str(y_pos*size+x_pos+1) + " " + str(y_pos*size+x_pos+2) + ' delete')
                # print(f"nodes {y_pos*size+x_pos+1} and {y_pos*size+x_pos+2}")
                cubit.cmd('create curve vertex ' + str(y_pos*size+x_pos+1) + " " + str(y_pos*size+x_pos+1+size) + ' delete')
                # print(f"nodes {y_pos*size+x_pos+1} and {y_pos*size+x_pos+1+size}")
                countcurves += 2
                if x_pos == (size-2):
                    cubit.cmd('create curve vertex ' + str(y_pos*size+x_pos+1+1) + " " + str(y_pos*size+x_pos+1+1+size) + ' delete')
                    # print(f"nodes {y_pos*size+x_pos+2} and {y_pos*size+x_pos+1+1+size}")
            else:
                cubit.cmd('create curve vertex ' + str(y_pos*size+x_pos+1) + " " + str(y_pos*size+x_pos+2) + ' delete')
                # print(f"nodes {y_pos*size+x_pos+1} and {y_pos*size+x_pos+2}")
  

        

    '''
    countcurves = 0
    for i in range(1,count+1):
    if i > 1 :
        if i % nstep == 0 :
        countcurves = countcurves + 1
        cubit.cmd('create curve spline vertex '+str(i-nstep+1) + ' to ' + str(i) + ' delete' )
    print '#done curves: '+str(countcurves)


    print '#creating skin surface...'
    cubit.cmd('create surface skin curve all')

    # cleans up
    cubit.cmd('merge all ')
    cubit.cmd('delete vertex all')
    cubit.cmd('delete curve all')

    '''
    # saves and exports surface
    # cubit file (uses ACIS file format)
    # cubit.cmd('save as "topo_2.cub" overwrite')
    #cubit.cmd('create surface skin curve all')
    cubit.cmd('create surface skin curve all')
    cubit.display_in_cubit()


if __name__ == '__main__':
    """Execution part of script."""

    #create_patch_test(False)
    rough_surface("sup5.dat")
