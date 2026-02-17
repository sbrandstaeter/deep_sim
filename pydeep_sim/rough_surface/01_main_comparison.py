import tamaas as tm
import matplotlib.pyplot as plt
import numpy as np
from tamaas.utils import load_path
import scipy
from count_switches import count_switches
import sys
import yaml
from subprocess import call
import os
import copy


def extract_float(line):
    a = line.rstrip("\n")  # return full line without newline
    b = a.split()          # split at any whitespace
    c = b[-1]              # take last element
    try:
        return float(c)
    except:
        return float(c[:-1])


path_to_mirco_IO = r'/home/a13ejabo/deep_sim/pydeep_sim/rough_surface/mirco_IO'
path_to_mirco_exe = r'/data/a13ejabo/MIRCO/build/mirco'

pattern_p = "Mean pressure is:"
pattern_a = "Effective contact area fraction is:"
pattern_t = "Elapsed time is:"

# Surface resolution (points pe rside)
n = 128

fname_mirco_input = os.path.join(
    path_to_mirco_IO, 'input_sup{}.yaml'.format(int(np.log2(n))))

fname_mirco_output = os.path.join(
    path_to_mirco_IO, 'output_sup{}.txt'.format(int(np.log2(n))))

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

# Store surface in MIRCO compliant format:
fname_surf = os.path.join(
    path_to_mirco_IO, 'sup{}.dat'.format(int(np.log2(n))))
np.savetxt(fname_surf,
           surface,
           fmt='%15.7e',
           delimiter=';',
           newline=';\n')

# Should be equal to 1 if normalized
rms_slope = tm.Statistics2D.computeSpectralRMSSlope(surface)

# Creates the model
model = tm.ModelFactory.createModel(tm.model_type.basic_2d, [L, L], [n, n])

# Equivalent non-periodic problem:
tm.ModelFactory.registerNonPeriodic(model, 'dcfft')

# Mechanical parameters
model.E = 1.0

# Initialize the solver
solver = tm.PolonskyKeerRey(model, surface, 1e-11)

# Solve equivalent non-periodic problem:
solver.setIntegralOperator('dcfft')

# Define load steps:
p_target = 0.1
loads = np.linspace(p_target/10, p_target/4, 10)

# Output from Tamaas:
A_raw_tamaas = np.empty((len(loads),))
A_cor_tamaas = np.empty_like(A_raw_tamaas)

Dmax_tamaas = np.empty_like(A_raw_tamaas)
Dmin_tamaas = np.empty_like(A_raw_tamaas)
Dmean_tamaas = np.empty_like(A_raw_tamaas)

# Output from MIRCO:
p0_mirco = np.empty_like(A_raw_tamaas)
A_raw_mirco = np.empty_like(A_raw_tamaas)
elapsed_time_mirco = np.empty_like(A_raw_tamaas)

# Solve for the given load path:
for i, model in enumerate(load_path(solver, loads)):

    solver.solve(loads[i])

    # To compute the true displacement (for non-periodic problem), one needs to re-evaluate the displacement:
    model.operators['dcfft'](model.traction, model.displacement)

    # Effective contact area (uncorrected):
    A_raw_tamaas[i] = dA*len(model.traction[model.traction > 0.0])

    # Perform area correction according to Yastrebov: !ATTENTION! This can not
    # be done with MIRCO at the corrent stage, so I will comment out the
    # modification that can be done in Tamaas.

    # M = count_switches(model.traction)
    # Sd = M*dx
    # A_cor_tamaas[i] = A_raw_tamaas[i] - (np.pi-1+np.log(2))/24*Sd*dx

    Dmin_tamaas[i] = np.min(model.displacement)
    Dmean_tamaas[i] = np.mean(model.displacement)
    Dmax_tamaas[i] = np.max(model.displacement)

    # Load input file:
    with open(fname_mirco_input, 'r') as f_pre:
        mirco_input_pre = yaml.safe_load(f_pre)

    # Update FFD in MIRCO input file:
    mirco_input_post = copy.deepcopy(mirco_input_pre)
    mirco_input_post["mirco_input"]["parameters"]["geometrical_parameters"]["Delta"] = float(
        1000*Dmax_tamaas[i])  # MIRCO works in mum.

    # Save updated input file:
    with open(fname_mirco_input, 'w') as f_post:
        yaml.dump(mirco_input_post, f_post, sort_keys=False)

    # Solve problem with MIRCO and store result in output file:
    with open(fname_mirco_output, 'w') as f:
        call(
            path_to_mirco_exe + ' ' + fname_mirco_input,
            shell=True,
            stdout=f)

    # Scope output file for desired quantities, i.e., effective contact area and
    # mean pressure:
    with open(fname_mirco_output, "r", encoding="utf-8") as file:
        for line in file:
            print(line)
            if pattern_p in line:
                p0_mirco[i] = extract_float(line)
            elif pattern_a in line:
                A_raw_mirco[i] = extract_float(line)
            elif pattern_t in line:
                elapsed_time_mirco[i] = extract_float(line)

    # Print info to screen:
    print(i, Dmax_tamaas[i], A_raw_tamaas[i], A_raw_mirco[i])

# Analytical models for comparison:
A_ab = np.sqrt(2*np.pi)*loads/rms_slope
A_pr = scipy.special.erf(np.sqrt(2)*loads/rms_slope)

################################################################################
# Plot results
################################################################################

fig, axs = plt.subplots(1, 1, figsize=(8, 6))
axs.plot(Dmax_tamaas, A_raw_mirco[:], '-^', label='mirco')
axs.plot(Dmax_tamaas, A_raw_tamaas[:], '-^', label='tamaas')
# axs.plot(loads/rms_slope, A_raw_tamaas[:]*100, '-^', label='tamaas')
# axs.plot(loads/rms_slope, A_raw_mirco[:]*100, '-^', label='tamaas')
# axs.set_ylabel('A (%)')
# axs.set_xlabel('p0')
# axs.legend(loc="upper right")
# axs.set_title('128 x 128')
plt.savefig('plot_cross_check_area.png')

fig, axs = plt.subplots(1, 1, figsize=(8, 6))
axs.plot(Dmax_tamaas, loads/np.max(loads), '-<', label='tamaas')
axs.plot(Dmax_tamaas, p0_mirco/np.max(p0_mirco), '->', label='mirco')
# axs.plot(loads/rms_slope, Dmax_tamaas/np.max(surface), '-<', label='n=128')
# axs.set_ylabel('D/z_max')
# axs.set_xlabel('p0')
# axs.grid(True)
# axs.legend(loc="upper right")
# axs.set_title('displacements')
plt.savefig('plot_cross_check_pressure.png')
