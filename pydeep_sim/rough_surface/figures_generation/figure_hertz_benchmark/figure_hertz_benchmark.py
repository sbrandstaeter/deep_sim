from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

R = 40.
E = 1.0
L = 1.0

df = pd.read_csv('hertz_benchmark.csv')
# print(df.head())

# print(rcParams.keys())

# -SIZES----------------------------------
fontsize = 7  # fontsize in (pt)
rcParams['axes.labelsize'] = fontsize
rcParams['axes.titlesize'] = fontsize
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize
rcParams['font.size'] = fontsize

# -FONTS-----------------------------------
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r"""\usepackage{siunitx}
       \sisetup{detect-all}
       \usepackage{helvet}
       \usepackage{sansmath}
       \sansmath
       """

# -MARKERS--------------------------------
rcParams['lines.markeredgecolor'] = r"k"
rcParams['lines.markeredgewidth'] = 0.75
rcParams['lines.markerfacecolor'] = r"r"
rcParams['lines.markersize'] = 5.0

# -LINES--------------------------------
rcParams['lines.linewidth'] = 1.0

# -PRINT PARAMS---------------------------
TEXTWIDTH = 507  # Textwidth in points
inchesperpt = 1.0/72

hor_size_pt = 0.5*TEXTWIDTH
hor_size = hor_size_pt*inchesperpt
ver_size = hor_size/1.618

fig, ax = plt.subplots(figsize=(hor_size, ver_size), dpi=300)
ax.set_facecolor('#EBEBEB')
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.grid(c='white')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.plot(df['Delta_an']/R, df['A_an']/L**2*100, 'b', label='Hertz')
ax.plot(df['Delta']/R, df['A']/L**2*100,
        marker='o',
        linestyle='',
        markevery=2,
        label='Tamaas')
ax.set_xlabel('$\\Delta/R$')
ax.set_ylabel('$A_\\mathrm{e}~(\\%)$')
ax.legend()
fig.tight_layout()
plt.savefig('benchmark_area.pdf')

fig, ax = plt.subplots(figsize=(hor_size, ver_size), dpi=300)
ax.set_facecolor('#EBEBEB')
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.grid(c='white')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.plot(df['Delta_an']/R, df['P_an']/(E*R**2), 'b', label='Hertz')
ax.plot(df['Delta']/R, df['P']/(E*R**2),
        marker='o',
        linestyle='',
        markevery=2,
        label='Tamaas')
ax.set_xlabel('$\\Delta/R$')
ax.set_ylabel('$F/(ER^2)$')
ax.set_xlim([1.0E-05, 6.2E-05])
ax.set_ylim([0.0, 6.5E-07])
ax.legend()
fig.tight_layout()
plt.savefig('benchmark_force.pdf')
