import numpy as np
from numpy.random import seed
from numpy import random as rnd
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import pandas as pd 

n = 5
size = 2**n+1
filename="sup5.dat"
z = np.loadtxt(fname=filename, usecols=range(size), delimiter=";")

print(z.mean())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0, size, 1)

X, Y = np.meshgrid(x, y)
zs = np.array(z)
print(zs.size)
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.view_init(azim=0, elev=90)

plt.show()