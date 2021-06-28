import numpy as np
from numpy import dtype, random as rnd


class RoughSurface():

    def __init__(self, output_dir, n , H, g0, n_iter):
        self.output_dir = output_dir
        self.n = n
        self.H = H
        self.g0 = g0
        self.n_iter = n_iter

    def generate_surface_RMD(self):
        '''
        creates the 2D surfaces using RMD (Random Midpoint Distribution)
        '''
        N = 2**self.n     
        z = np.zeros([N+1,N+1])

        alpha = 1 / np.sqrt(0.09)

        D = N
        d = N//2

        for _ in range(self.n):
            alpha=alpha/np.sqrt(2)**self.H
            
            for j in range(d,N-d+1,D):
                for k in range(d,N-d+1,D):
                    z[j,k] =  (z[j+d,k+d]+z[j+d,k-d]+z[j-d,k+d]+z[j-d,k-d])/4+alpha*rnd.randn()
            
            alpha=alpha/np.sqrt(2)**self.H
            
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

        # scalefactor = g0/(np.max(z)-np.mean(z))
        # z = z*scalefactor
        # z = z-(np.min(z))
        z = self.g0*(z-np.min(z))/(np.max(z)-np.min(z)); # (scaling between 0 and g0)


        full_path = self.output_dir + "/surface_" + str(self.n_iter) + ".dat"
        np.savetxt(full_path, z, delimiter=';', fmt="%15.5e")
        
        return full_path