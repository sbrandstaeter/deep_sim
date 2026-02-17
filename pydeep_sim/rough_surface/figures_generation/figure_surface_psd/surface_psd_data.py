import sys
import numpy as np
import tamaas as tm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tamaas.utils import radial_average

# Parameters:
i = 2
q0 = [1, 4, 16]
q1 = [1, 4, 16]
q2 = [64, 128, 128]


# Create spectrum object
spectrum = tm.Isopowerlaw2D()

# Set spectrum parameters
spectrum.q0 = q0[i]
spectrum.q1 = q1[i]
spectrum.q2 = q2[i]
spectrum.hurst = 0.8

n = 512
generator = tm.SurfaceGeneratorFilter2D([n, n])
generator.spectrum = spectrum
generator.random_seed = 1

surface = generator.buildSurface()
surface /= tm.Statistics2D.computeSpectralRMSSlope(surface)
# print(tm.Statistics2D.computeSpectralRMSSlope(surface))

x = np.linspace(0.0, 1.0, n)
y = np.linspace(0.0, 1.0, n)
X, Y = np.meshgrid(x, y)

beta = 2*(spectrum.hurst+1)
sigma = np.std(surface)

C0 = sigma**2*(2-beta)/(2*np.pi*(spectrum.q2**(2-beta)-spectrum.q1**(2-beta)))

psd = tm.Statistics2D.computePowerSpectrum(surface)
psd = np.fft.fftshift(psd, axes=0).real
qx = np.fft.fftshift(np.fft.fftfreq(surface.shape[0], d=1 / surface.shape[0]))
qy = np.fft.rfftfreq(surface.shape[1], d=1 / surface.shape[1])

r = np.linspace(1, qy.max() - 1, 128)
theta = np.linspace(0, np.pi, 30)

psd_rad = radial_average(qx, qy, psd, r, theta,
                         method="nearest", endpoint=True)

psd_an = C0*(r/spectrum.q1)**-beta

# Export data for analytic and numerical PSD
data_psd = np.vstack((r, psd_rad/spectrum.q1**-beta, psd_an)).T

# Export data for height histogram
counts, bins = np.histogram(surface, 512)
data_hist = np.vstack((np.pad(counts, (0, 1), mode='constant'), bins)).T

# Export data for surface heights
data_surf = np.vstack((X.ravel(), Y.ravel(), surface.ravel())).T

np.savetxt(
    'surface{}_PSD.csv'.format(i),
    data_psd,
    fmt="%.5e",
    delimiter=",",
    header="r, psd_num, psd_an",
    comments="")

np.savetxt(
    'surface{}_xyz.csv'.format(i),
    data_surf,
    fmt="%.5e",
    delimiter=",",
    header="x, y, z",
    comments="")

np.savetxt(
    'data_histogram_surface{}.csv'.format(i),
    data_hist,
    fmt="%.5e",
    delimiter=",",
    header="",
    comments="")
