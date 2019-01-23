import sys
sys.path.append('../')

from schmactions import schmactions

import numpy as np

import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import pickle

from scipy.interpolate import interp1d
from scipy.optimize import minimize

import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
name_list = ['thin', 'thick']

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

init_pos = [8, 0, 0] * u.kpc
init_vel_list = [[0, -190, 10] * u.km/u.s,
                 [0, -190, 50] * u.km/u.s,
                 [0, -190, 190] * u.km/u.s]
clist = [tb_c[7], tb_c[8], tb_c[0]]

# set up figure
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

for name, init_vel, c in zip(name_list, init_vel_list, clist):
    s = schmactions(init_pos, init_vel)
    zout = pickle.load(open('zout_'+name+'.p', 'rb'))
    res = pickle.load(open('true_res_'+name+'.p', 'rb'))
    J0, J1, J2 = res['actions'].to_value(u.kpc*u.km/u.s)

    zact = np.array([ s.extract_actions(r) for r in zout['act_result'] ]) 
    zup = np.percentile(zact, 95, axis=1)
    zlow = np.percentile(zact, 5, axis=1)
    dz = (zup - zlow)

    offlist = np.array(zout['offset_list'])[:,2]
    dJzJz = dz[:,2]/J2

    offlist_target = zoffset_gen(offlist, dJzJz)

    ax.plot(mc_list, offlist_target*1000, c=c, label=name)

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

