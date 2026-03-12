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

hurst = 0.6069856853131297
q1 = 1
q2 = 32
random_seed = 1085218879

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
) = generate_surface_and_solve_pressure_driven_eff_area_individual_load_steps(
    q1=int(q1),
    q2=int(q2),
    hurst=hurst,
    n=1024,
    L=1.0,
    random_seed=int(random_seed),
    p_target=0.4,
    num_load_steps=10,
    periodic=False,
    solver_tolerance=solver_tolerance,
    scale_factor_surface=scale_factor_surface,
    solve_contact_problem=True,
    random_load_steps=True,
)
print(f"solved in {time.time() - start_time}s.")
