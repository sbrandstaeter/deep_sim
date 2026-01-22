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

scale_pressure_rms_slope = [True, False]

# surfaces = []
# A_raws = []
# A_cors = []
#

# surfaces = []
# A_raws = []
# A_cors = []
# loads = []
# rms_slopes = []
# Dmins = []
# Dmeans = []
# Dmaxs = []
# run_times = []
# # for scale_pressue in scale_pressure_rms_slope:
#     print("#########################################")
#     print(f"Solving with tolerance: {solver_tolerance}")
#     start_time = time.time()
#     (
#         surface,
#         A_raw,
#         A_cor,
#         load,
#         rms_slope,
#         Dmin,
#         Dmean,
#         Dmax,
#         run_time,
#     ) = generate_surface_and_solve_pressure_driven_eff_area(
#         q1=int(q1),
#         q2=int(q2),
#         hurst=hurst,
#         n=512,
#         L=1.0,  # Surface lateral size
#         random_seed=int(random_seed),
#         p_target=0.3,
#         num_load_steps=20,
#         periodic=False,
#         solver_tolerance=solver_tolerance,
#         scale_pressure=scale_pressue,
#     )
#     print(f"solved in {time.time() - start_time}s.")
#     surfaces.append(surface)
#     A_raws.append(A_raw)
#     A_cors.append(A_cor)
#     loads.append(load)
#     rms_slopes.append(rms_slope)
#     Dmins.append(Dmin)
#     Dmeans.append(Dmean)
#     Dmaxs.append(Dmax)
#     run_times.append(run_time)
#
# np.savez(
#     "tamaas_compare_rms_scaling.npz",
#     surfaces=np.array(surfaces),
#     A_raws=np.array(A_raws),
#     A_cors=np.array(A_cors),
#     loads=np.array(loads),
#     rms_slopes=np.array(rms_slopes),
#     Dmins=np.array(Dmins),
#     Dmeans=np.array(Dmeans),
#     Dmaxs=np.array(Dmaxs),
#     run_times=np.array(run_times),
# )

data = np.load("tamaas_compare_rms_scaling.npz")

surfaces = data["surfaces"]
A_raws = data["A_raws"]
A_cors = data["A_cors"]
loads = data["loads"]
rms_slopes = data["rms_slopes"]
Dmins = data["Dmins"]
Dmeans = data["Dmeans"]
Dmaxs = data["Dmaxs"]
run_times = data["run_times"]

# optional: close the file handle (good practice)
data.close()
colors = pc.qualitative.Plotly  # or Dark24, D3, Set2, etc.

fig = go.Figure()

for i, scale_pressure in enumerate(scale_pressure_rms_slope):
    load = loads
    if scale_pressure:
        load /= rms_slopes[i]
    fig.add_trace(
        go.Scatter(
            x=load,
            y=A_cors[i],
            mode="lines",
            name=f"{scale_pressure}",
            line=dict(
                color=colors[i % len(colors)],
                dash="solid" if scale_pressure else "dash",
            ),
        )
    )

fig.show()
fig.write_html("tamaas_compare_rms_scaling.html")

print("Hello")
