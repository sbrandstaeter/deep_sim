from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from numpy import dtype, random as rnd
from scipy.stats import norm, kurtosis
from scipy.stats import skew


class RoughSurface:

    def __init__(
        self,
        output_dir,
        file_tail,
        Resolution,
        Hurst,
        InitialTopologyStdDeviation,
        n_iter,
        LateralLength,
    ):
        self.output_dir = output_dir
        self.file_tail = file_tail
        self.Resolution = Resolution
        self.Hurst = Hurst
        self.InitialTopologyStdDeviation = InitialTopologyStdDeviation
        self.n_iter = n_iter
        self.LateralLength = LateralLength

    def generate_surface_RMD(self):
        """
        creates the 2D surfaces using RMD (Random Midpoint Distribution)
        """
        N = 2**self.Resolution
        z = np.zeros([N + 1, N + 1])

        alpha = self.InitialTopologyStdDeviation / np.power(np.sqrt(2), self.Hurst)

        D = N
        d = N // 2

        for _ in range(self.Resolution):
            alpha = alpha / np.power(np.sqrt(2), self.Hurst)

            for j in range(d, N - d + 1, D):
                for k in range(d, N - d + 1, D):
                    z[j, k] = (
                        z[j + d, k + d]
                        + z[j + d, k - d]
                        + z[j - d, k + d]
                        + z[j - d, k - d]
                    ) / 4 + alpha * rnd.randn()

            alpha = alpha / np.power(np.sqrt(2), self.Hurst)

            for j in range(d, N - d + 1, D):
                z[j, 0] = (
                    z[j + d, 0] + z[j - d, 0] + z[j, d]
                ) / 3 + alpha * rnd.randn()
                z[j, N] = (
                    z[j + d, N] + z[j - d, N] + z[j, N - d]
                ) / 3 + alpha * rnd.randn()
                z[0, j] = (
                    z[0, j + d] + z[0, j - d] + z[d, j]
                ) / 3 + alpha * rnd.randn()
                z[N, j] = (
                    z[N, j + d] + z[N, j - d] + z[N - d, j]
                ) / 3 + alpha * rnd.randn()

            for j in range(d, N - d + 1, D):
                for k in range(D, N - d + 1, D):
                    z[j, k] = (
                        z[j, k + d] + z[j, k - d] + z[j + d, k] + z[j - d, k]
                    ) / 4 + alpha * rnd.randn()

            for j in range(D, N - d + 1, D):
                for k in range(d, N - d + 1, D):
                    z[j, k] = (
                        z[j, k + d] + z[j, k - d] + z[j + d, k] + z[j - d, k]
                    ) / 4 + alpha * rnd.randn()

            D = D // 2
            d = d // 2

        # scalefactor = g0/(np.max(z)-np.mean(z))
        # z = z*scalefactor
        z = z - (np.min(z))

        # z = self.g0*(z-np.min(z))/(np.max(z)-np.min(z)); # (scaling between 0 and g0)

        full_path = Path(self.output_dir) / ("topology_" + self.file_tail + ".dat")
        np.savetxt(full_path, z, delimiter=";", fmt="%15.5e")

        return full_path


def random_postprocess(rough_surface, lateral_length):

    # import the rough surface
    z = rough_surface
    n_heights = z.shape[0]

    # the element size
    ele_length = lateral_length / n_heights

    # -------------------------------------------------------------------
    # Compute the profile statistics of the peaks: slopes, maxima (2D) heights and  curvatures
    # -------------------------------------------------------------------

    # calculate the slopes of each height
    slope_x, slope_y = np.gradient(z, ele_length)

    # discard the boundaries
    slope_x_boudary = slope_x[1:-1, 1:-1]
    slope_y_boudary = slope_y[1:-1, 1:-1]

    # rms slope (surface gradient) based on first-order finite difference
    slope_x_fd = np.diff(z, axis=0) / ele_length
    slope_y_fd = np.diff(z, axis=1) / ele_length

    slope_squared = slope_x_fd[:, :-1] ** 2 + slope_y_fd[:-1, :] ** 2
    rms_slope_fd = np.sqrt(np.mean(slope_squared))

    # print(slope_x_boudary.shape)
    # print(slope_y_boudary.shape)
    # print(slope_x_fd.shape)
    # print(slope_y_fd.shape)
    # print(slope_squared.shape)
    # print(rms_slope_fd)
    # print(np.sqrt(np.mean(slope_x_boudary**2 + slope_y_boudary**2)))

    # import tamaas as tm
    # print(tm.Statistics2D.computeFDRMSSlope(z))
    # print(tm.Statistics2D.computeSpectralRMSSlope(z))

    # evaluate the 2D maxima (peaks) curvatures
    n_peaks = 0
    curv_peak = []
    z_peak = []
    for j in range(1, n_heights - 1):
        for i in range(1, n_heights - 1):
            if z[i, j] > z[i, j - 1] and z[i, j] > z[i, j + 1]:
                n_peaks += 1
                curv_peak.append(
                    -(z[i, j + 1] - 2 * z[i, j] + z[i, j - 1]) / (ele_length**2)
                )
                z_peak.append(z[i, j])

    # statistics of the slopes and peaks
    # statistics of z
    m0 = np.std(z, ddof=1)

    # statistics profile slopes
    rms_slopex = np.std(slope_x_boudary, ddof=1)
    rms_slopey = np.std(slope_y_boudary, ddof=1)

    m2x = rms_slopex**2
    m2y = rms_slopey**2

    # statistics of the heights of the peaks
    mean_z_peaks = np.mean(z_peak)  # 1
    rms_z_peaks = np.std(z_peak, ddof=1)  # 2
    ks_z_peaks = kurtosis(z_peak, fisher=False)  # 3
    sk_z_peaks = skew(z_peak)  # 4

    # statistics of the curvatures of the peaks
    mean_curv_peaks = np.mean(curv_peak)  # 5
    rms_curv_peaks = np.std(curv_peak, ddof=1)
    ks_curv_peaks = kurtosis(curv_peak)  # 6
    sk_curv_peaks = skew(curv_peak)  # 7

    m4 = rms_curv_peaks**2

    # density of peaks
    density_peaks = n_peaks / (n_heights * n_heights)

    alfa_x = m0 * m4 / m2x**2
    alfa_y = m0 * m4 / m2y**2

    # -------------------------------------------------------------------
    # Compute the asperity statistics: (3D maxima) heights and curvatures
    # -------------------------------------------------------------------

    # create the mesh
    x_lin = np.linspace(lateral_length / n_heights, lateral_length, n_heights)
    y_lin = np.linspace(lateral_length / n_heights, lateral_length, n_heights)

    y, x = np.meshgrid(x_lin, y_lin)

    curv_asperity_x = np.zeros((n_heights - 2, n_heights - 2))
    curv_asperity_y = np.zeros((n_heights - 2, n_heights - 2))

    # calculate the asperity curvatures
    for i in range(1, n_heights - 1):
        for j in range(1, n_heights - 1):
            if z[i, j] > z[i, j - 1] and z[i, j] > z[i, j + 1]:
                curv_asperity_x[i - 1, j - 1] = (
                    -2
                    * (
                        -ele_length * z[i, j - 1]
                        + 2 * ele_length * z[i, j]
                        - ele_length * z[i, j + 1]
                    )
                    / (
                        -ele_length * y[j, j - 1] ** 2
                        + 2 * ele_length * y[j, j] ** 2
                        - ele_length * y[j, j + 1] ** 2
                    )
                )

    for i in range(1, n_heights - 1):
        for j in range(1, n_heights - 1):
            if z[i, j] > z[i - 1, j] and z[i, j] > z[i + 1, j]:
                curv_asperity_y[i - 1, j - 1] = (
                    -2
                    * (
                        -ele_length * z[i - 1, j]
                        + 2 * ele_length * z[i, j]
                        - ele_length * z[i + 1, j]
                    )
                    / (
                        -ele_length * x[j - 1, j] ** 2
                        + 2 * ele_length * x[j, j] ** 2
                        - ele_length * x[j + 1, j] ** 2
                    )
                )

    mask_vector = curv_asperity_x * curv_asperity_y
    mask_crierion = mask_vector != 0
    # calculate the overall curvatures of asperity
    curv_asperity = np.sqrt(mask_vector[mask_crierion])
    z_without_border = z[1:-1, 1:-1]
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
    density_asperities = height_asperity.shape[0] / (n_heights * n_heights)

    # collect the statistical results in the statistical_properties variable
    statistical_properties = {}

    statistical_properties["mean_z_peaks"] = mean_z_peaks  # mean
    statistical_properties["rms_z_peaks"] = rms_z_peaks  # root_mean_squared peaks
    statistical_properties["rms_slope_peaks"] = (
        rms_slope_fd  # root_mean_squared slope (rms surface gradient) peaks
    )
    statistical_properties["ks_z_peaks"] = ks_z_peaks  # kurtosis
    statistical_properties["sk_z_peaks"] = sk_z_peaks  # skewness
    statistical_properties["mean_curv_peaks"] = mean_curv_peaks  # mean
    statistical_properties["ks_curv_peaks"] = ks_curv_peaks  # kurtosis
    statistical_properties["sk_curv_peaks"] = sk_curv_peaks  # skewness
    statistical_properties["dn_peaks"] = density_peaks
    statistical_properties["alfa_x"] = alfa_x
    statistical_properties["alfa_y"] = alfa_y

    statistical_properties["mean_z_asp"] = mean_z_asperities
    statistical_properties["rms_z_asp"] = rms_z_asperities
    statistical_properties["ks_z_asp"] = ks_z_asperities
    statistical_properties["sk_z_asp"] = sk_z_asperities
    statistical_properties["mean_curv_asp"] = mean_curv_asperities
    statistical_properties["rms_curv_asp"] = rms_curv_asperities
    statistical_properties["ks_curv_asp"] = ks_curv_asperities
    statistical_properties["sk_curv_asp"] = sk_curv_asperities
    statistical_properties["dns_asp"] = density_asperities

    # -------------------------------------------------------------------
    # Compute the statistics of the rough surface itself
    # -------------------------------------------------------------------
    statistical_properties["z_mean"] = z.mean()
    statistical_properties["z_max"] = z.max()
    statistical_properties["z_rms"] = z.std()

    return statistical_properties


def plot_surface(rough_surface, lateral_length, output_dir):

    # Create grid coordinates
    x = np.linspace(0, lateral_length, rough_surface.shape[0])
    y = np.linspace(0, lateral_length, rough_surface.shape[1])
    X, Y = np.meshgrid(x, y)

    # Make 3D surface plot
    fig = go.Figure(data=[go.Surface(z=rough_surface, x=X, y=Y, colorscale="Viridis")])

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="height",
        ),
        title="3D Surface Plot",
    )

    # Save as HTML
    fig.write_html(output_dir / "surface_plot.html")
