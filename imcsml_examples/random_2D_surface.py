import numpy as np
from numpy.random import seed
from numpy import random as rnd
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt

def random_2D_surface(*args):
    '''
        define the n and H
    '''
    n = args[0]
    H = args[1]

#    rnd = np.random.uniform(1,100)
#    seed(95) # make it comment otherwise there will be same data 
    N = 2**n
  
    
    #initialize z
    z = np.zeros([N+1,N+1])  #dimension is N*N

    alpha = 1 / np.sqrt(0.09)

    D = N
    d = N//2

    for _ in range(n):
        alpha=alpha/np.sqrt(2)**H
        
        for j in range(d,N-d+1,D):
            for k in range(d,N-d+1,D):
                z[j,k] =  (z[j+d,k+d]+z[j+d,k-d]+z[j-d,k+d]+z[j-d,k-d])/4+alpha*rnd.randn()
        
        alpha=alpha/np.sqrt(2)**H
        
        for j in range(d,N-d+1,D):
            z[j,0]=(z[j+d,0]+z[j-d,0]+z[j,d])/3+alpha*rnd.randn()
            z[j,N]=(z[j+d,N]+z[j-d,N]+z[j,N-d])/3+alpha*rnd.randn()
            z[0,j]=(z[0,j+d]+z[0,j-d]+z[d,j])/3+alpha*rnd.randn()
            z[N,j]=(z[N,j+d]+z[N,j-d]+z[N-d,j])/3+alpha*rnd.randn()
  
        for j in range(d,N-d+1,D):
            for k in range(D,N-d+1,D):
                z[j,k]=(z[j,k+d]+z[j,k-d]+z[j+d,k]+z[j-d,k])/4+alpha*rnd.randn()
        
        for j in range(D,N-d+1,D):
            for k in range(d,N-d+1,D):
                z[j,k]=(z[j,k+d]+z[j,k-d]+z[j+d,k]+z[j-d,k])/4+alpha*rnd.randn()


        D = D//2
        d = d//2 

    #print(z)

    zref = 25 # reference for the scaling, former value = 25
    scalefactor = zref/(np.max(z)-np.mean(z))
    z = z*scalefactor
    z = z-(np.min(z))
    np.savetxt('data.csv', z, delimiter=';')

    print(f"mean value is {np.mean(z)}")
    #print(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 2**n+1, 1)

    X, Y = np.meshgrid(x, y)
    zs = np.array(z)
    print(zs.size)
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    #ax.view_init(azim=0, elev=90)

    plt.show()


if __name__ == "__main__":
    random_2D_surface(5,0.75)