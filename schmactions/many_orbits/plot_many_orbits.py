import sys
sys.path.append('../')

from schmactions import schmactions

import astropy.units as u
import pickle

import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
import numpy as np

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

xmin = 0
xmax = 500

y0min = 0
y0max = 100
y1min = 0
y1max = 100
y2min = 0
y2max = 200

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

name_list = ['thin', 'thick', 'halo']

fig, ax = plt.subplots(2, 3, figsize=(7, 4))
init_pos = [8, 0, 0] * u.kpc
init_vel_list = [[0, -190, 10] * u.km/u.s,
                 [0, -190, 50] * u.km/u.s,
                 [0, -190, 190] * u.km/u.s]

clist = [tb_c[7], tb_c[8], tb_c[0]]

for name, init_vel, c in zip(name_list, init_vel_list, clist):
    zout = pickle.load(open('zout_'+name+'.p', 'rb'))
    xout = pickle.load(open('xout_'+name+'.p', 'rb'))
    res = pickle.load(open('true_res_'+name+'.p', 'rb'))
    J0, J1, J2 = res['actions'].to_value(u.kpc*u.km/u.s)

    print(name, J0, J1, J2)

    s = schmactions(init_pos, init_vel)

    zact = np.array([ s.extract_actions(r) for r in zout['act_result'] ]) 
    xact = np.array([ s.extract_actions(r) for r in xout['act_result'] ]) 

    zoffset = np.array(zout['offset_list'])[:,2]
    xoffset = np.array(xout['offset_list'])[:,0]

    zup = np.percentile(zact, 95, axis=1)
    zlow = np.percentile(zact, 5, axis=1)
    xup = np.percentile(xact, 95, axis=1)
    xlow = np.percentile(xact, 5, axis=1)

    dz = (zup - zlow)
    dz[:,0] /= J0
    dz[:,1] /= J1
    dz[:,2] /= J2

    dx = (xup - xlow)
    dx[:,0] /= J0
    dx[:,1] /= J1
    dx[:,2] /= J2
    
    for x,o,d in zip(ax, (zoffset, xoffset), (dz, dx)):
        # get keys corresponding to 4 sigmaclip
        # k0, k1, k2 = sclip(a)
        x[0].plot(o, 100*d[:,0], c=c)
        x[1].plot(o, -100*d[:,1], c=c, label=name)
        x[2].plot(o, 100*d[:,2], c=c, label=name)
    
        # x[0].plot(t, np.full(len(t), J0), c=tb_c[4], ls='dashed')
        # x[1].plot(t, np.full(len(t), J1), c=tb_c[4], ls='dashed')
        # x[2].plot(t, np.full(len(t), J2), c=tb_c[4], ls='dashed')
    
        # set limits on plots
        x[0].set_ylim(y0min, y0max)
        x[1].set_ylim(y1min, y1max)
        x[2].set_ylim(y2min, y2max)
        for xx in x:
            xx.set_xlim(xmin, xmax)

for x in ax[0]:
    x.set_xlabel(r'$z\,\text{offset}\,[\,\text{pc}\,]$')
    x.set_xticks(np.arange(xmin,xmax,100), minor=True)

for x in ax[1]:
    x.set_xlabel(r'$x\,\text{offset}\,[\,\text{pc}\,]$')
    x.set_xticks(np.arange(xmin,xmax,100), minor=True)

for x in ax[:,0]:
    x.set_ylabel(r'$\Delta J_r/J_r\,[\,\%\,]$')
for x in ax[:,1]:
    x.set_ylabel(r'$\Delta L_z/L_z\,[\,\%\,]$')
for x in ax[:,2]:
    x.set_ylabel(r'$\Delta J_z/J_z\,[\,\%\,]$')

ax[0][1].legend(frameon=False, title='orbit')

fig.tight_layout()
plt.savefig('schmactions_many_orbits.pdf')
plt.close()

# fig, ax = plt.subplots(1,1, figsize=(3,3))

# k0, k1, k2 = sclip(zact)
# ax.hist(zact[:,2][k2], bins=np.arange(histmin, histmax, 0.25), lw=2,
#         edgecolor=tb_c[4], fc='none', histtype='stepfilled')
# ax.axvline(x=J2, color=tb_c[4], ls='dashed', lw=2)
# ax.set_xlabel(r'$J_{z,\text{obs}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
# ax.set_ylabel(r'$\text{count}$')

# fig.tight_layout()
# plt.savefig('schmactions_Jz_hist.pdf')

