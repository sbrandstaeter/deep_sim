import numpy as np
from numpy import dtype, random as rnd
from scipy.stats import norm, kurtosis
from scipy.stats import skew


class RoughSurface():

    def __init__(self, output_dir, n , H, g0, n_iter, lato):
        self.output_dir = output_dir
        self.n = n
        self.H = H
        self.g0 = g0
        self.n_iter = n_iter
        self.lato = lato

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
    
    def random_postprocess(self, statistical_properties):
        
        # the rough surface path
        full_path = self.output_dir + "/surface_" + str(self.n_iter) + ".dat"

        # the number of heights
        n_heights = 2**self.n + 1

        # import the rough surface
        z = np.loadtxt(full_path, delimiter=";",usecols=range(n_heights))

        # the element size
        ele_length = self.lato/n_heights

        # -------------------------------------------------------------------
        # Compute the profile statistics of the peaks: slopes, maxima (2D) heights and  curvatures
        # -------------------------------------------------------------------

        # calculate the slopes of each height
        slope_x, slope_y = np.gradient(z,ele_length)

        # discard the boundaries
        slope_x_boudary = slope_x[1:-1,1:-1]
        slope_y_boudary = slope_y[1:-1,1:-1]

        # evaluate the 2D maxima (peaks) curvatures
        n_peaks = 0
        curv_peak = []
        z_peak = []
        for j in range(1,n_heights-1):
            for i in range(1,n_heights-1):
                if z[i, j] > z[i, j-1] and z[i,j]>z[i,j+1]:
                    n_peaks += 1
                    curv_peak.append(-(z[i,j+1]-2*z[i,j]+z[i,j-1])/(ele_length**2))
                    z_peak.append(z[i,j])

        # statistics of the slopes and peaks
        # statistics of z
        m0 = np.std(z, ddof =1)

        # statistics profile slopes
        rms_slopex = np.std(slope_x_boudary ,ddof =1)
        rms_slopey = np.std(slope_y_boudary, ddof =1)

        m2x = rms_slopex**2
        m2y = rms_slopey**2


        # statistics of the heights of the peaks
        mean_z_peaks = np.mean(z_peak) #1
        rms_z_peaks = np.std(z_peak, ddof =1) #2
        ks_z_peaks = kurtosis(z_peak, fisher=False) #3
        sk_z_peaks = skew(z_peak) #4

        # statistics of the curvatures of the peaks
        mean_curv_peaks = np.mean(curv_peak) #5
        rms_curv_peaks = np.std(curv_peak, ddof = 1)
        ks_curv_peaks = kurtosis(curv_peak) #6
        sk_curv_peaks = skew(curv_peak) #7

        m4 = rms_curv_peaks**2

        # density of peaks
        density_peaks = n_peaks/(n_heights*n_heights) 

        alfa_x = m0*m4/m2x**2
        alfa_y = m0*m4/m2y**2

        # collect the statistical results in the statistical_properties variable
        statistical_properties["mean_z_peaks"].append(mean_z_peaks) # mean
        statistical_properties["rms_z_peaks"].append(rms_z_peaks) # root_mean_square
        statistical_properties["ks_z_peaks"].append(ks_z_peaks) # kurtosis
        statistical_properties["sk_z_peaks"].append(sk_z_peaks) # skewness
        statistical_properties["mean_curv_peaks"].append(mean_curv_peaks) # mean
        statistical_properties["ks_curv_peaks"].append(ks_curv_peaks) # kurtosis
        statistical_properties["sk_curv_peaks"].append(sk_curv_peaks) # skewness
        statistical_properties["dn_peaks"].append(density_peaks) 
        statistical_properties["alfa_x"].append(alfa_x)
        statistical_properties["alfa_y"].append(alfa_y)

        # -------------------------------------------------------------------
        # Compute the asperity statistics: (3D maxima) heights and curvatures
        # -------------------------------------------------------------------

        # create the mesh 
        x_lin = np.linspace(self.lato/n_heights,self.lato,n_heights)
        y_lin = np.linspace(self.lato/n_heights,self.lato,n_heights)

        y, x = np.meshgrid(x_lin, y_lin)

        curv_asperity_x = np.zeros((n_heights-2,n_heights-2))
        curv_asperity_y = np.zeros((n_heights-2,n_heights-2))

        # calculate the asperity curvatures
        for i in range(1,n_heights-1):
            for j in range(1,n_heights-1):
                if z[i,j]>z[i,j-1] and z[i,j]>z[i,j+1]:
                    curv_asperity_x[i-1,j-1]=-2*(-ele_length*z[i,j-1]+2*ele_length*z[i,j]-ele_length*z[i,j+1])/ \
                    (-ele_length*y[j,j-1]**2+2*ele_length*y[j,j]**2-ele_length*y[j,j+1]**2)

        for i in range(1,n_heights-1):
            for j in range(1,n_heights-1):
                if z[i,j]>z[i-1,j] and z[i,j]>z[i+1,j]:
                    curv_asperity_y[i-1,j-1]=-2*(-ele_length*z[i-1,j]+2*ele_length*z[i,j]-ele_length*z[i+1,j])/ \
                    (-ele_length*x[j-1,j]**2+2*ele_length*x[j,j]**2-ele_length*x[j+1,j]**2)


        mask_vector = curv_asperity_x*curv_asperity_y
        mask_crierion = (mask_vector!= 0)
        # calculate the overall curvatures of asperity 
        curv_asperity = np.sqrt(mask_vector[mask_crierion])
        z_without_border = z[1:-1,1:-1]
        # calculate the overall heights of asperity 
        height_asperity = z_without_border[mask_crierion]

        # statistics of asperity (3D maxima) heights
        mean_z_asperities = np.mean(height_asperity)
        rms_z_asperities = np.std(height_asperity, ddof=1)
        ks_z_asperities = kurtosis(height_asperity)
        sk_z_asperities = skew(height_asperity)

        # statistics of asperity (3D maxima) curvatures
        mean_curv_asperities = np.mean(curv_asperity)
        rms_curv_asperities = np.std(curv_asperity, ddof=1)
        ks_curv_asperities = kurtosis(curv_asperity, fisher=False)
        sk_curv_asperities = skew(curv_asperity)

        # the density of the asperities
        density_asperities = height_asperity.shape[0]/(n_heights*n_heights)

       # collect the statistical results in the statistical_properties variable
        statistical_properties["mean_z_asp"].append(mean_z_asperities)
        statistical_properties["rms_z_asp"].append(rms_z_asperities)
        statistical_properties["ks_z_asp"].append(ks_z_asperities)
        statistical_properties["sk_z_asp"].append(sk_z_asperities)
        statistical_properties["mean_curv_asp"].append(mean_curv_asperities)
        statistical_properties["rms_curv_asp"].append(rms_curv_asperities)
        statistical_properties["ks_curv_asp"].append(ks_curv_asperities)
        statistical_properties["sk_curv_asp"].append(sk_curv_asperities)
        statistical_properties["dns_asp"].append(density_asperities)

        # -------------------------------------------------------------------
        # Compute the statistics of the rough surface itself
        # -------------------------------------------------------------------
        statistical_properties["z_mean"].append(z.mean())
        statistical_properties["z_rms"].append(z.std())
        
        return  statistical_properties