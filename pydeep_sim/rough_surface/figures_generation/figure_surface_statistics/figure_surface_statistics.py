import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams

df = pd.read_csv('surface.0512x0512.xyz.csv')

indx = np.argmin(np.abs(df['x'].unique()-40.0))
indy = np.argmin(np.abs(df['y'].unique()-40.0))

x_bar = df['x'].unique()[indx]
y_bar = df['y'].unique()[indy]

z_mean = np.mean(df['z'])

x = df[df['y'] == y_bar]['x']
y = df[df['x'] == x_bar]['y']
zx = np.array(df[df['y'] == y_bar]['z'])
zy = np.array(df[df['x'] == x_bar]['z'])

profile_data = np.vstack((x, y, zx, zy)).T
np.savetxt('profile_{}.csv'.format(indx),
           profile_data,
           header='x,y,zx,zy',
           comments='',
           delimiter=',')

# -SIZES----------------------------------
fontsize = 6  # fontsize in (pt)
rcParams['axes.labelsize'] = fontsize
rcParams['axes.titlesize'] = fontsize
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize
rcParams['font.size'] = fontsize

# -FONTS-----------------------------------
# rcParams['text.usetex'] = True
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# -LEGEND-----------------------------------
rcParams['legend.loc'] = 'upper center'

# -PRINT PARAMS---------------------------
TEXTWIDTH = 507  # Width in points
inchesperpt = 1.0 / 72

hor_size_pt = 0.5*TEXTWIDTH
hor_size = hor_size_pt*inchesperpt
ver_size = hor_size/2

fig, ax = plt.subplots(2, 1, figsize=(hor_size, ver_size), dpi=300)

ax[0].hlines(0.0, 0.0, 1.0, linestyle='-.', linewidth=0.5, color='k')
ax[0].plot(x/100, (zx-z_mean)/100, c='b', linewidth=0.75)
# Give plot a gray background like ggplot.
ax[0].set_facecolor('#EBEBEB')
# Remove border around plot.
[ax[0].spines[side].set_visible(False) for side in ax[0].spines]
ax[0].grid(c='white')
ax[0].set_xticklabels([])
ax[0].xaxis.set_ticks_position('none')
ax[0].yaxis.set_ticks_position('none')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([-0.025, 0.025])
# ax[0].set_xlabel('$y/L$')
ax[0].set_ylabel('$z/L$')

ax[1].hlines(0.0, 0.0, 1.0, linestyle='-.', linewidth=0.5, color='k')
ax[1].plot(y/100, (zy-z_mean)/100, c='r', linewidth=0.75)
# Give plot a gray background like ggplot.
ax[1].set_facecolor('#EBEBEB')
# Remove border around plot.
[ax[1].spines[side].set_visible(False) for side in ax[1].spines]
ax[1].grid(c='white')
ax[1].xaxis.set_ticks_position('none')
ax[1].yaxis.set_ticks_position('none')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([-0.025, 0.025])
ax[1].set_xlabel('$(x,y)/L$')
ax[1].set_ylabel('$z/L$')
# leg = ax[].legend(ncols=2)
fig.tight_layout()
fig.savefig('results.pdf')

# Profiles have to be equal at intersection point:
assert (np.abs(zy[204]-zx[204]) < 1.0E-15)
