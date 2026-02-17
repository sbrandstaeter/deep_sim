import tamaas as tm
import matplotlib.pyplot as plt
import numpy as np
from tamaas.utils import load_path
import scipy
from count_switches import count_switches

# Surface resolution (points pe rside)
n = 512

# Surface lateral size
L = 1.0

# Grid cell area
dx = L/n
dA = dx**2

# Surface generator
sg = tm.SurfaceGeneratorFilter2D([n, n])
sg.random_seed = 1

# Define spectrum
sg.spectrum = tm.Isopowerlaw2D()

# Roll-off wavenumber (defines ratio between lateral size of the surface and longest wavelength. if lambda_0=2*pi/q0 ~ L the surface is not Gaussian)
sg.spectrum.q0 = 16
# Lowest wavenumber (defines longest wavelength of the surface as lambda_1 = 2*pi/q1)
sg.spectrum.q1 = 16
# Highest wavenumber (defines shortest wavelength of the surface as lambda_2 = 2*pi/q2)
sg.spectrum.q2 = 64
# Defines slope of the spectrum
sg.spectrum.hurst = 0.8

# Generates the surface given the spectrum
surface = sg.buildSurface()

rms_slope = tm.Statistics2D.computeSpectralRMSSlope(surface)

# Comment not to normalize the RMSSLope
surface /= rms_slope

# Should be equal to 1 if normalized
rms_slope = tm.Statistics2D.computeSpectralRMSSlope(surface)

# Creates the model
model = tm.ModelFactory.createModel(tm.model_type.basic_2d, [L, L], [n, n])

# Uncomment to solve equivalent non-periodic problem:
tm.ModelFactory.registerNonPeriodic(model, 'dcfft')

# Mechanical parameters
model.E = 1.0

# Initialize the solver
solver = tm.PolonskyKeerRey(model, surface, 1e-10)

# Uncomment to solve equivalent non-periodic problem:
solver.setIntegralOperator('dcfft')

# Define load steps:
p_target = 0.1

# Solve for given load step:
solver.solve(p_target)

# To compute the true displacement (for non-periodic problem), one needs to re-evaluate the displacement
model.operators['dcfft'](model.traction, model.displacement)

# Effective contact area (uncorrected)
A_raw = dA*len(model.traction[model.traction > 0.0])

# Perform correction of the area according to Yastrebov:
M = count_switches(model.traction)
Sd = M*dx
A_cor = A_raw - (np.pi-1+np.log(2))/24*Sd*dx

# Analytical models for benchmarking (only reliable for periodic problems!)
A_ab = np.sqrt(2*np.pi)*p_target/rms_slope
A_pr = scipy.special.erf(np.sqrt(2)*p_target/rms_slope)

# Minimum and maximum displacement
Dmin = np.min(model.displacement)
Dmax = np.max(model.displacement)

# Reference datum for displacement is set in correspondence of mean elevation of the surface,
# so that only relative displacements can be evaluated for periodic problems
#print(A_pr, A_cor,  A_raw, A_ab)
print(Dmin, Dmax)
print(np.min(surface), np.mean(surface), np.max(surface))
