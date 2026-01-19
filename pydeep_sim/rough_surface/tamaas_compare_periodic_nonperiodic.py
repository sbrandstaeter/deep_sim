import time

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from pydeep_sim.rough_surface.tamaas_load_path import (
    generate_surface_and_solve_pressure_driven_eff_area,
)

hurst = 0.8
q1 = 4
q2 = 128
random_seed = 1
solver_tolerance = 1e-09

periodicity = [True, False]

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
for periodic in periodicity:
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
        p_target=0.3,
        num_load_steps=20,
        periodic=periodic,
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
    "compare_periodicity.npz",
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

if len(surfaces) > 1 and np.testing.assert_allclose(surfaces[0], surfaces[1]):
    print("solved for same surfaces")


colors = pc.qualitative.Plotly  # or Dark24, D3, Set2, etc.

fig = go.Figure()

for i, periodic in enumerate(periodicity):
    fig.add_trace(
        go.Scatter(
            x=loads[i] / rms_slopes[i],
            y=A_cors[i],
            mode="lines",
            name=f"{periodic}",
            line=dict(color=colors[i % len(colors)]),
        )
    )

fig.show()
fig.write_html("compare_periodicity.html")
