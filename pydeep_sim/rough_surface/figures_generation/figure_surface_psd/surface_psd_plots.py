from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

# Parameters:
i = 2
q0 = [1, 4, 16]
q1 = [1, 4, 16]
q2 = [64, 128, 128]

data_psd = np.loadtxt(
    'data/surface{}_PSD.csv'.format(i),
    delimiter=",",
    skiprows=1)
 
r = data_psd[:, 0]
psd_num = data_psd[:, 1]
psd_an = data_psd[:, 2]

data_hist = np.loadtxt(
    'data/data_histogram_surface{}.csv'.format(i),
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
rcParams['lines.markeredgewidth'] = 0.5
rcParams['lines.markerfacecolor'] = r"r"
rcParams['lines.markersize'] = 3.0
print(rcParams.keys())

# -LINES--------------------------------
rcParams['lines.linewidth'] = 0.5

# -SIZES----------------------------------
font_size = 6
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

########################################################################

fig, ax = plt.subplots(1, 1, figsize=(hor_size, ver_size), dpi=300)

ax.loglog(r, psd_an, '-.k')
ax.loglog(r[8:], psd_num[8:], '-or', markevery=2)

ax.set_xlim([1,256])
ax.set_ylim([1.0E-15, 1.0E-00])

# Give plot a gray background like ggplot:
ax.set_facecolor('#EBEBEB')

# Remove border around plot:
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.grid(c='white')

# X-ticks:
num = int(np.log2(q2[i])-np.log2(q1[i])+1)
xtick_positions = np.logspace(0,8, base=2, num=9)
ytick_positions = np.logspace(-15,0, base=10, num=6)

ax.xaxis.set_major_locator(FixedLocator(xtick_positions))
ax.xaxis.set_minor_locator(FixedLocator([]))

ax.yaxis.set_major_locator(FixedLocator(ytick_positions))
ax.yaxis.set_minor_locator(FixedLocator([]))

xtick_labels = [str(int(val)) for val in xtick_positions]
ytick_labels = ['{:10.1e}'.format(val) for val in ytick_positions]

ax.set_xticklabels(xtick_labels,)
ax.set_yticklabels(ytick_labels,)

ax.set_xlabel('$q\\,(\\si{\\meter^{-1}})$')
ax.set_ylabel('$\\Phi_\\mathrm{p}/\\Phi_0$')

ax.tick_params(width=0.1)

fig.tight_layout()
fig.savefig('surface{}_psd.pdf'.format(i), format='pdf')

########################################################################

"""
plt.figure(2)
plt.hist(bins[:-1], bins, weights=counts)
plt.savefig('test_hist.png')
"""

