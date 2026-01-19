import time

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from pydeep_sim.rough_surface.tamaas_load_path import (
    generate_surface_and_solve_pressure_driven_eff_area,
)

experiment_name = "tamaas_points_7"
job_id = 38
hurst = 0.6
q1 = int(16.0)
q2 = int(128.0)
random_seed = int(1891650887.0)
solver_tolerances = [1e-09, 1e-11]

surfaces = []
A_raws = []
A_cors = []


surfaces = []
A_raws = []
A_cors = []
loads = []
rms_slopes = []
Dmins = []
Dmeans = []
Dmaxs = []
run_times = []
for solver_tolerance in solver_tolerances:
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
    ) = generate_surface_and_solve_pressure_driven_eff_area(
        q1=int(q1),
        q2=int(q2),
        hurst=hurst,
        n=512,
        L=1.0,  # Surface lateral size
        random_seed=int(random_seed),
        p_target=0.2,
        num_load_steps=50,
        periodic=False,
        solver_tolerance=solver_tolerance,
    )
    print(f"solved in {time.time() - start_time}s.")
    surfaces.append(surface)
    A_raws.append(A_raw)
    A_cors.append(A_cor)
    loads.append(load)
    rms_slopes.append(rms_slope)
    Dmins.append(Dmin)
    Dmeans.append(Dmean)
    Dmaxs.append(Dmax)
    run_times.append(run_time)

np.savez(
    "compare_tolerances.npz",
    surfaces=np.array(surfaces),
    A_raws=np.array(A_raws),
    A_cors=np.array(A_cors),
    loads=np.array(load),
    rms_slopes=np.array(rms_slopes),
    Dmins=np.array(Dmins),
    Dmeans=np.array(Dmeans),
    Dmaxs=np.array(Dmaxs),
    run_times=np.array(run_times),
)

if np.testing.assert_allclose(surfaces[0], surfaces[1]):
    print("solved for same surfaces")


colors = pc.qualitative.Plotly  # or Dark24, D3, Set2, etc.

fig = go.Figure()

for i, solver_tolerance in enumerate(solver_tolerances):
    fig.add_trace(
        go.Scatter(
            x=loads[i] / rms_slopes[i],
            y=A_cors[i],
            mode="lines",
            name=f"tol:{solver_tolerance}",
            line=dict(color=colors[i % len(colors)]),
        )
    )

fig.show()
fig.write_html("compare_tolerances.html")
