import numpy as np

import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp

from scipy.interpolate import interp1d
from scipy.optimize import minimize

import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

Rsolar = 8.2

#cluster min, max mass
mc_min = 100
mc_max = 10000
mc_list = np.linspace(mc_min, mc_max, 1000)

# find mass enclosed within Rsolar
mw = gp.MilkyWayPotential()
q = gd.PhaseSpacePosition(pos=[Rsolar, 0, 0] * u.kpc, vel=[0, 0, 0] * u.km/u.s)
M_enclosed = float(mw.mass_enclosed(q).to_value(u.Msun))

dJzJz_target = np.power(mc_list / M_enclosed, 1/3)

# load in results
result_z = np.load('result_z.p')
init_act_list = np.load('init_act_list.p')

# set up figure
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

for result_orbit, action_orbit, c in zip(result_z, init_act_list, tb_c):
    offlist = result_orbit[0]
    dJz = result_orbit[1][:,2]
    Jz = action_orbit[2]
    dJzJz = dJz/Jz

def zoffset_gen(offlist, dJzJz):
    interp = interp1d(offlist, dJzJz)

    def to_minimize(off, target):
        return np.abs(interp(float(off)) - target)

    offlist_target = []
    for this_dJzJz in dJzJz_target:
        this_off = float(minimize(to_minimize, 0, args=(this_dJzJz,)).x)
        offlist_target.append(this_off)
    offlist_target = np.array(offlist_target)
    return offlist_target

# set up figure
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

for result_orbit, action_orbit, c in zip(result_z, init_act_list, tb_c):
    offlist = result_orbit[0]
    dJz = result_orbit[1][:,2]
    Jz = action_orbit[2]
    dJzJz = dJz/Jz

    offlist_target = zoffset_gen(offlist, dJzJz)

    ax.plot(mc_list, offlist_target*1000, c=c, label="{0:0.1f}".format(Jz))

ax.set_xlabel(r'$m_c\,[\,M_{\odot}\,]$')
ax.set_ylabel(r'$z\,\text{offset}\,[\,\text{pc}\,]$')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim((10, 2000))
ax.set_xlim((mc_min, mc_max))

ax.legend(frameon=False, title=r'$J_{z,\text{true}}$')

fig.tight_layout()
fig.savefig('cluster_offset.pdf')
plt.close()
