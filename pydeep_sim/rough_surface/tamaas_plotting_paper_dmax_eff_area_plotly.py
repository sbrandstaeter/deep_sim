import pandas as pd
import numpy as np
import plotly.express as px

experiment_name = "tamaas_points_nonperiodic_3"

combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")

surfaces_mean_std = np.load(f"{experiment_name}_surfaces_stds_mean.npy")


fig1 = px.scatter(
    combined_data_df,
    x="dmax",
    y="run_times",
    # color=np.ceil(combined_data_df["ids"] / 50),
    # color_discrete_sequence=px.colors.qualitative.Set2,
    labels={
        "dmax": "normalized Δ [-]",  # Plotly doesn't reliably render full LaTeX in axis titles
        "run_times": "Wall clock time [s]",
        "ids": "ID",
    },
)

# ---- print-ready layout ----
fig1.update_traces(
    marker=dict(size=9, opacity=0.85, line=dict(width=0.6, color="rgba(0,0,0,0.35)")),
)

latex_font = "Latin Modern Roman"  # , Computer Modern Serif, Times New Roman"
fig1.update_layout(
    template="simple_white",
    width=900,  # ~ single-column wide; bump to 1200 for two-column
    height=600,
    font=dict(family=latex_font, size=24),
    title=None,  # typically cleaner for print; add caption in paper instead
    margin=dict(l=80, r=30, t=30, b=70),
    legend=dict(
        font=dict(size=22),
        title_font=dict(size=24),
        title_text="ID",
        orientation="v",
        x=1.02,
        xanchor="left",
        y=1.0,
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        itemsizing="constant",
    ),
)

fig1.update_xaxes(
    autorange=False,
    range=[0, 0.26],
    tickmode="linear",
    tick0=0,
    dtick=0.05,
    title_font=dict(size=26, family=latex_font),
    tickfont=dict(size=22, family=latex_font),
    title_standoff=12,
    ticks="outside",
    ticklen=6,
    tickwidth=1,
    showline=True,
    linewidth=1,
    mirror=True,
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(0,0,0,0.15)",
    zeroline=False,
)

fig1.update_yaxes(
    autorange=False,
    range=[0, 40],
    tickmode="linear",
    tick0=0,
    dtick=5,
    title_font=dict(size=24, family=latex_font),
    tickfont=dict(size=22, family=latex_font),
    title_standoff=12,
    ticks="outside",
    ticklen=6,
    tickwidth=1,
    showline=True,
    linewidth=1,
    mirror=True,
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(0,0,0,0.15)",
    zeroline=False,
)

fig1.show()

# interactive version
fig1.write_html(f"{experiment_name}_scatter_dmax_run_time.html", include_plotlyjs="cdn")

# ---- vector export for print (recommended) ----
# pip install -U kaleido
fig1.write_image(
    f"{experiment_name}_scatter_dmax_run_time.svg"
)  # vector, great for papers
fig1.write_image(f"{experiment_name}_scatter_dmax_run_time.pdf")  # vector PDF
# fig1.write_image(f"{experiment_name}_scatter_dmax_run_time.png", scale=3)  # hi-res raster fallback
