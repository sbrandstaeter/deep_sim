import time

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import tamaas as tm
import matplotlib.pyplot as plt
from tamaas.utils import load_path
import scipy

from count_switches import count_switches
from pydeep_sim.rough_surface.tamaas_load_path import (
    generate_surface_and_solve_pressure_driven_eff_area_load_path,
    generate_surface_and_solve_pressure_driven_eff_area_individual_load_steps,
)


hurst = 0.8
q1 = 4
q2 = 128
random_seed = 1
solver_tolerance = 1e-09

scale_pressure_rms_slope = [True, False]

print("#########################################")
print(f"Solving with tolerance: {solver_tolerance}")
start_time = time.time()
(
    surface,
    A_raw,
    A_cor,
    load,
    rms_slope,
    Dmin,
    Dmean,
    Dmax,
    run_time,
) = generate_surface_and_solve_pressure_driven_eff_area_load_path(
    q1=int(q1),
    q2=int(q2),
    hurst=hurst,
    n=512,
    L=1.0,  # Surface lateral size
    random_seed=int(random_seed),
    p_target=0.3,
    num_load_steps=20,
    periodic=False,
    solver_tolerance=solver_tolerance,
    scale_factor_surface=1.0,
    solve_contact_problem=True,
)
print(f"solved in {time.time() - start_time}s.")

start_time = time.time()
(
    surface2,
    A_raw2,
    A_cor2,
    load2,
    rms_slope2,
    Dmin2,
    Dmean2,
    Dmax2,
    run_time2,
) = generate_surface_and_solve_pressure_driven_eff_area_individual_load_steps(
    q1=int(q1),
    q2=int(q2),
    hurst=hurst,
    n=512,
    L=1.0,  # Surface lateral size
    random_seed=int(random_seed),
    p_target=0.3,
    num_load_steps=20,
    periodic=False,
    solver_tolerance=solver_tolerance,
    scale_factor_surface=1.0,
    solve_contact_problem=True,
)
print(f"solved in {time.time() - start_time}s.")

np.savez(
    "tamaas_compare_run_times.npz",
    surfaces=np.array([surface, surface2]),
    A_raws=np.array([A_raw, A_raw2]),
    A_cors=np.array([A_cor, A_cor2]),
    loads=np.array([load, load2]),
    rms_slopes=np.array([rms_slope, rms_slope2]),
    Dmins=np.array([Dmin, Dmin2]),
    Dmeans=np.array([Dmean, Dmean2]),
    Dmaxs=np.array([Dmax, Dmax2]),
    run_times=np.array([run_time, run_time2]),
)
