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
from pydeep_sim.rough_surface.tamaas_queens_driver import scale_factor


hurst = 0.7
q1 = 1
q2 = 32
random_seed = 26012026
solver_tolerance = 1e-09

print("#########################################")
print(f"Solving with tolerance: {solver_tolerance}")
start_time = time.time()
scale_factor_surface = scale_factor(q1, q2)
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
    L=1.0,
    random_seed=int(random_seed),
    p_target=4.5,
    num_load_steps=100,
    periodic=False,
    solver_tolerance=solver_tolerance,
    scale_factor_surface=scale_factor_surface,
    solve_contact_problem=True,
)
print(f"solved in {time.time() - start_time}s.")

np.savez(
    "tamaas_compute_eff_area_over_pressure_for_run_time.npz",
    q1=np.array(q1),
    q2=np.array(q2),
    hurst=np.array(hurst),
    seed=np.array(random_seed),
    surface=np.array(surface),
    A_raw=np.array(A_raw),
    A_cor=np.array(A_cor),
    loads=np.array(load),
    rms_slope=np.array(rms_slope),
    Dmin=np.array(Dmin),
    Dmean=np.array(Dmean),
    Dmax=np.array(Dmax),
    run_time=np.array(run_time),
)
