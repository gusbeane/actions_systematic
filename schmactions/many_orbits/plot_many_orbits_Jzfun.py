import sys
sys.path.append('../')

from schmactions import schmactions

import astropy.units as u
import pickle

import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

xmin = 0
xmax = 100

ymin = 0
ymax = 300

histmin = 15
histmax = 30

def sclip(a, s=4):
    _, low0, high0 = sigmaclip(a[:,0], low=s, high=s)
    _, low1, high1 = sigmaclip(a[:,1], low=s, high=s)
    _, low2, high2 = sigmaclip(a[:,2], low=s, high=s)
    a0bool = np.logical_and(a[:,0] > low0, a[:,0] < high0)
    a1bool = np.logical_and(a[:,1] > low1, a[:,1] < high1)
    a2bool = np.logical_and(a[:,2] > low2, a[:,2] < high2)
    k0 = np.where(a0bool)[0]
    k1 = np.where(a1bool)[0]
    k2 = np.where(a2bool)[0]
    return k0, k1, k2

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
init_pos = [8, 0, 0] * u.kpc
init_vel = [0, -190, 10] * u.km/u.s

s = schmactions(init_pos, init_vel)

out = pickle.load(open('dJz_fun_of_Jz.p', 'rb'))

to_plot = []
for act, r in out:
    try:
        Jz = act[2]
        up = np.percentile(r, 95, axis=1)[2]
        low = np.percentile(r, 5, axis=1)[2]
        to_plot.append([Jz, 100*(up-low)/Jz])
    except:
        to_plot.append([np.nan, np.nan])
to_plot = np.array(to_plot)

nanbool1 = np.logical_not(np.isnan(to_plot[:,0]))
nanbool2 = np.logical_not(np.isnan(to_plot[:,0]))
nanbool = np.logical_or(nanbool1, nanbool2)

arbbool = to_plot[:,1] < 1E7
totbool = np.logical_and(nanbool, arbbool)

# ignore some numerical artifacts
totbool[0] = False
totbool[75] = False
totbool[80] = False
keys = np.where(totbool)[0]

ax.plot(to_plot[:,0][keys], to_plot[:,1][keys], c=tb_c[-1])
ax.set_yscale('log')

# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])

ax.set_xlabel(r'$J_z\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
ax.set_ylabel(r'$\Delta J_z/J_z$')

fig.tight_layout()
plt.savefig('schmactions_many_orbits_Jz_fun.pdf')
plt.close()
