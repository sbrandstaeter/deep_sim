from matplotlib import rcParams
from matplotlib import pyplot as plt
import numpy as np

# Parameters:
i = 2
q0 = [1, 4, 16]
q1 = [1, 4, 16]
q2 = [64, 128, 128]

data_psd = np.loadtxt(
    'surface{}_PSD.csv'.format(i),
    delimiter=",",
    skiprows=1)

r = data_psd[:, 0]
psd_num = data_psd[:, 1]
psd_an = data_psd[:, 2]

data_hist = np.loadtxt(
    'data_histogram_surface{}.csv'.format(i),
    delimiter=",",
    skiprows=1)

counts = data_hist[:-1, 0]
bins = data_hist[:, 1]

################################################################################


# -PRINT PARAMS---------------------------
TEXTWIDTH = 507  # From package layout in latex document
inchesperpt = 1.0 / 72.27
golden_ratio = (np.sqrt(5) - 1.0) / 2.0

portion = 1/3  # Portion of textwidth occupied by figure
hor_size = TEXTWIDTH*portion*inchesperpt
ver_size = hor_size*golden_ratio

# -MARKERS--------------------------------
rcParams['lines.markeredgecolor'] = r"k"
rcParams['lines.markeredgewidth'] = 0.75
rcParams['lines.markerfacecolor'] = r"r"
rcParams['lines.markersize'] = 5.0

# -LINES--------------------------------
rcParams['lines.linewidth'] = 1.0

# -SIZES----------------------------------
font_size = 7
rcParams['axes.labelsize'] = font_size
rcParams['axes.titlesize'] = font_size
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.labelsize'] = font_size
rcParams['ytick.labelsize'] = font_size
rcParams['legend.fontsize'] = font_size
rcParams['font.size'] = font_size

# -FONTS-----------------------------------
"""

rcParams['font.serif'] = ['Computer Modern Roman']
"""
rcParams['text.usetex'] = True
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Helvetica'
rcParams['text.latex.preamble'] = \
    r"""
        \usepackage{siunitx}
        \usepackage{helvet}
        \usepackage{sfmath}
    """

"""plt.figure(1)
plt.loglog(r, psd_num)
plt.loglog(r, psd_an)
plt.savefig('test_psd.png')"""

"""plt.figure(2)
plt.hist(bins[:-1], bins, weights=counts)
plt.savefig('test_hist.png')"""


fig, ax = plt.subplots(1, 1, figsize=(hor_size, ver_size), dpi=300)

ax.plot(r[8:], psd_an[8:], '-.k')
ax.plot(r[8:], psd_num[8:], '-or')
ax.loglog()

# Give plot a gray background like ggplot.
ax.set_facecolor('#EBEBEB')

# Remove border around plot.
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.grid(c='white')
# ax.set_xlim([8, 512])
ax.set_ylim([1.0E-12, 1.0E-03])
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_xlabel('$q\\,(\\si{\\meter^{-1}})$')
ax.set_ylabel('$\\Phi_\\mathrm{p}/\\Phi_0$')
# leg = ax[].legend(ncols=2)
ax.set_xticks(
    np.logspace(np.log2(q1[i]),
                np.log2(q2[i]), base=2, num=8),
    labels=['1/4', '1/2', '1', '32', '64', '128', '256', '512']) #TODO: represent as q/qr

fig.tight_layout()
fig.savefig('surface{}_psd.png'.format(i), format='png')
