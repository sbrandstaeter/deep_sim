from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

R = 40.
E = 1.0
rms_slope = 1
L = 1.0

df1 = pd.read_csv('rough_contact_area.csv')
# print(df1.head())

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

hor_size_pt = int(TEXTWIDTH)/4
hor_size = hor_size_pt*inchesperpt
ver_size = hor_size

fig, ax = plt.subplots(figsize=(hor_size, ver_size), dpi=300)
ax.set_facecolor('#EBEBEB')
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.grid(c='white')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.plot(df1['loads']/E, df1['A_ab']/L**2, 'g', label='Asperity model')
ax.plot(df1['loads']/E, df1['A_pr']/L**2, 'b', label='Persson model')
ax.plot(df1['loads']/E, df1['A']/L**2, 'r', linestyle='-.', label='Tamaas')
ax.plot(df1['loads'][9]/E, df1['A'][9]/L**2, markerfacecolor='cyan', marker='*',markersize='6')
ax.plot(df1['loads'][29]/E, df1['A'][29]/L**2, markerfacecolor='cyan', marker='*',markersize='6')
ax.set_xlabel('$p_0/E^\\ast$')
ax.set_ylabel('$A_\\mathrm{e}/A_0~(\\%)$')
ax.legend()
fig.tight_layout()
plt.savefig('rough_contact_area.pdf')
