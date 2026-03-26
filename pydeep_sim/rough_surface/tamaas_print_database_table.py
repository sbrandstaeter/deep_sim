#!/usr/bin/env python3
"""
Generate the LaTeX table from a pandas DataFrame.

Usage:
    python make_table.py data.csv

The script:
- reads the CSV file into a pandas DataFrame
- computes min/max for the selected columns
- formats them in scientific notation with 2 significant figures
- prints the LaTeX table to stdout
"""
from typing import List, Dict, Any

import pandas as pd


# ----------------------------------------------------------------------
# Table specification
# ----------------------------------------------------------------------
ROWS: List[Dict[str, Any]] = [
    {
        "type": "Load",
        "name": r"Far-field displacement",
        "symbol": r"$\Delta$",
        "column": "dmax",
    },
    {
        "type": r"\begin{tabular}[c]{c}Statistical \\ parameters\end{tabular}",
        "name": r"Mean of peaks",
        "symbol": r"$\bar{z}_{\text{p}}$",
        "column": "mean_z_peaks",
    },
    {
        "type": None,
        "name": r"Root mean square of peaks",
        "symbol": r"$\text{RMS}[z_{\text{p}}]$",
        "column": "rms_z_peaks",
    },
    {
        "type": None,
        "name": r"Kurtosis of peaks",
        "symbol": r"$\text{K}[z_{\text{p}}]$",
        "column": "ks_z_peaks",
    },
    {
        "type": None,
        "name": r"Skewness of peaks",
        "symbol": r"$\text{Sk}[z_{\text{p}}]$",
        "column": "sk_z_peaks",
    },
    {
        "type": None,
        "name": r"Density of peaks",
        "symbol": r"$\rho_{\text{p}}$",
        "column": "dn_peaks",
    },
    {
        "type": None,
        "name": r"Mean of curvature of peaks",
        "symbol": r"$\bar{\kappa}_{\text{p}}$",
        "column": "mean_curv_peaks",
    },
    {
        "type": None,
        "name": r"Kurtosis of curvature of peaks",
        "symbol": r"$\text{K}[\kappa_{\text{p}}]$",
        "column": "ks_curv_peaks",
    },
    {
        "type": None,
        "name": r"Skewness of curvature of peaks",
        "symbol": r"$\text{Sk}[\kappa_{\text{p}}]$",
        "column": "sk_curv_peaks",
    },
    {
        "type": None,
        "name": r"Bandwidth parameter in x direction",
        "symbol": r"$\alpha_x$",
        "column": "alfa_x",
    },
    {
        "type": None,
        "name": r"Bandwidth parameter in y direction",
        "symbol": r"$\alpha_y$",
        "column": "alfa_y",
    },
    {
        "type": None,
        "name": r"Mean of asperities",
        "symbol": r"$\bar{z}_{\text{a}}$",
        "column": "mean_z_asp",
    },
    {
        "type": None,
        "name": r"Root mean square of asperities",
        "symbol": r"$\text{RMS}[z_{\text{a}}]$",
        "column": "rms_z_asp",
    },
    {
        "type": None,
        "name": r"Kurtosis of asperities",
        "symbol": r"$\text{K}[z_{\text{a}}]$",
        "column": "ks_z_asp",
    },
    {
        "type": None,
        "name": r"Skewness of asperities",
        "symbol": r"$\text{Sk}[z_{\text{a}}]$",
        "column": "sk_z_asp",
    },
    {
        "type": None,
        "name": r"Density of asperities",
        "symbol": r"$\rho_{\text{a}}$",
        "column": "dns_asp",
    },
    {
        "type": None,
        "name": r"Mean of curvature of asperities",
        "symbol": r"$\bar{\kappa}_{\text{a}}$",
        "column": "mean_curv_asp",
    },
    {
        "type": None,
        "name": r"Root mean square of curvature of asperities",
        "symbol": r"$\text{RMS}[\kappa_{\text{a}}]$",
        "column": "rms_curv_asp",
    },
    {
        "type": None,
        "name": r"Kurtosis of curvature of asperities",
        "symbol": r"$\text{K}[\kappa_{\text{a}}]$",
        "column": "ks_curv_asp",
    },
    {
        "type": None,
        "name": r"Skewness of curvature of asperities",
        "symbol": r"$\text{Sk}[\kappa_{\text{a}}]$",
        "column": "sk_curv_asp",
    },
    # {
    #     "type": None,
    #     "name": r"Mean of surface height",
    #     "symbol": r"$\bar{z}$",
    #     "column": "z_mean",
    # },
    {
        "type": None,
        "name": r"Max.\ of surface height",
        "symbol": r"$z^{\text{max}}$",
        "column": "z_max",
    },
    {
        "type": None,
        "name": r"Root mean square of surface height",
        "symbol": r"$\text{RMS}[z]$",
        "column": "z_rms",
    },
    {
        "type": None,
        "name": r"Root mean square of slope",
        "symbol": r"$\text{RMS}[\nabla z]$",
        "column": "rms_slope",
    },
    {
        "type": "Target quantity",
        "name": r"Effective contact area",
        "symbol": r"$\EffectArea$",
        "column": "eff_area",
    },
]


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def format_sci(x: float) -> str:
    """
    Return a string suitable for siunitx S columns, e.g.
    0.0674 -> '6.7e-2'
    45.0   -> '4.5e1'
    5.0    -> '5.0e0'
    """
    return f"{x:.5e}"#.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-").replace("e00", "e0")


def row_line(type_cell: str, name: str, symbol: str, min_str: str, max_str: str) -> str:
    return f"{type_cell}\n& {name} & {symbol} & {min_str} & {max_str} \\\\"


# ----------------------------------------------------------------------
# Main LaTeX generation
# ----------------------------------------------------------------------
def generate_table(df: pd.DataFrame) -> str:
    stats_rows = [r for r in ROWS if r["type"] is None]
    stats_count = len(stats_rows)

    lines: List[str] = []

    lines.append(r"{")
    lines.append(r"\sisetup{")
    lines.append(r"  round-mode               = figures,")
    lines.append(r"  round-precision          = 2,")
    lines.append(r"  scientific-notation      = true,")
    lines.append(r"  retain-zero-exponent     = true,")
    lines.append(r"  % exponent-product       = \cdot,")
    lines.append(r"}")
    lines.append("")
    lines.append(r"\begin{table}[thbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{\TODOT{The ranges need to be updated}\TODOS{Overview of the generated database including the input quantities, namely the displacement load and the statistical surface parameters, as well as the output target quantity, the effective contact area. Each row lists a quantity together with its corresponding symbol and the range of values represented in the database.}}"
    )
    lines.append(r"    \label{tab:list_of_params}")
    lines.append(r"    \renewcommand{\arraystretch}{1.1}")
    lines.append(r"    \setlength{\tabcolsep}{6pt}")
    lines.append(r"    \begin{tabular}{")
    lines.append(r"        c")
    lines.append(r"        l")
    lines.append(r"        c")
    lines.append(r"        S[table-format=-1.1e1]")
    lines.append(r"        S[table-format=-1.1e1]")
    lines.append(r"    }")
    lines.append(r"        \toprule")
    lines.append(
        r"        \multirow{2}{*}{\textbf{Type}} & \multirow{2}{*}{\textbf{Name}} & \multirow{2}{*}{\textbf{Symbol}} & \multicolumn{2}{c}{\textbf{Range}} \\"
    )
    lines.append(r"        \cmidrule(lr){4-5}")
    lines.append(r"         &  &  & {Min} & {Max} \\")
    lines.append(r"        \midrule")
    lines.append("")

    # Load row
    load_row = next(r for r in ROWS if r["type"] == "Load")
    load_min = format_sci(df[load_row["column"]].min())
    load_max = format_sci(df[load_row["column"]].max())
    lines.append(
        row_line(r"\begin{tabular}[c]{c}Boundary \\ condition\end{tabular}", load_row["name"], load_row["symbol"], load_min, load_max)
    )
    lines.append(r"\midrule")
    lines.append("")

    # Statistical block
    first_stat = True
    for spec in stats_rows:
        col = spec["column"]
        min_str = format_sci(df[col].min())
        max_str = format_sci(df[col].max())

        if first_stat:
            type_cell = rf"\multirow{{{stats_count}}}{{*}}{{{spec['type'] or r'\begin{tabular}[c]{c}Statistical \\ parameters\end{tabular}'}}}"
            # use explicit desired label
            type_cell = rf"\multirow{{{stats_count}}}{{*}}{{\begin{{tabular}}[c]{{c}}Statistical \\ parameters\end{{tabular}}}}"
            first_stat = False
        else:
            type_cell = ""

        lines.append(row_line(type_cell, spec["name"], spec["symbol"], min_str, max_str))

    lines.append(r"\midrule")
    lines.append("")

    # Target row
    target_row = next(r for r in ROWS if r["type"] == "Target quantity")
    target_min = format_sci(df[target_row["column"]].min())
    target_max = format_sci(df[target_row["column"]].max())
    lines.append(
        row_line("Target quantity", target_row["name"], target_row["symbol"], target_min, target_max)
    )
    lines.append("")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"}")

    # Indent row lines inside the tabular for readability
    out_lines = []
    for line in lines:
        if line.startswith(("&", "Load", r"\multirow", "Target quantity", r"\midrule", r"\bottomrule", r"\cmidrule", r"\toprule")):
            out_lines.append("        " + line)
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


def main() -> None:
    experiment_name = "tamaas_points_nonperiodic_3"
    df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")
    experiment_name_test_data = "tamaas_points_nonperiodic_3_test_data"

    df_test_data = pd.read_parquet(f"{experiment_name_test_data}.parquet", engine="pyarrow")

    df_concat = pd.concat([df, df_test_data])


    min_max = df_concat.agg(['min', 'max']).T
    print(min_max)
    print()
    print("#############################################################################")
    print()

    stats = pd.concat(
        [
            df.agg(["min", "max"]).T.add_suffix("_grid"),
            df_test_data.agg(["min", "max"]).T.add_suffix("_rand"),
        ],
        axis=1,
    )

    stats = stats[["min_grid", "min_rand", "max_grid", "max_rand"]]
    print(stats)
    print()
    print("#############################################################################")
    print()

    latex = generate_table(df_concat)
    print(latex)


if __name__ == "__main__":
    main()