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
xmax = 1000

y0min = 20
y0max = 40
y1min = -1800
y1max = -1000
y2min = 10
y2max = 30

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

zout = pickle.load(open('zout.p', 'rb'))
xout = pickle.load(open('xout.p', 'rb'))
res = pickle.load(open('true_res.p', 'rb'))
J0, J1, J2 = res['actions'].to_value(u.kpc*u.km/u.s)

init_pos = [8, 0, 0] * u.kpc
init_vel = [0, -190, 50] * u.km/u.s

s = schmactions(init_pos, init_vel)

zact = s.extract_actions(zout)
xact = s.extract_actions(xout)

ztime = s.extract_time(zout)
xtime = s.extract_time(xout)

fig, ax = plt.subplots(2, 3, figsize=(7, 4))

for x,t,a in zip(ax, (ztime, xtime), (zact, xact)):
    # get keys corresponding to 4 sigmaclip
    k0, k1, k2 = sclip(a)
    x[0].plot(t[k0], a[:,0][k0], c=tb_c[4])
    x[1].plot(t[k1], a[:,1][k1], c=tb_c[4])
    x[2].plot(t[k2], a[:,2][k2], c=tb_c[4])

    x[0].plot(t, np.full(len(t), J0), c=tb_c[4], ls='dashed')
    x[1].plot(t, np.full(len(t), J1), c=tb_c[4], ls='dashed')
    x[2].plot(t, np.full(len(t), J2), c=tb_c[4], ls='dashed')

    # set limits on plots
    x[0].set_ylim(y0min, y0max)
    x[1].set_ylim(y1min, y1max)
    x[2].set_ylim(y2min, y2max)
    for xx in x:
        xx.set_xlim(xmin, xmax)


for x in ax[1]:
    x.set_xlabel(r'$t\,[\,\text{Myr}\,]$')
    x.set_xticks(np.arange(0,1000,100), minor=True)

for x in ax[:,0]:
    x.set_ylabel(r'$J_r\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
for x in ax[:,1]:
    x.set_ylabel(r'$L_z\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
for x in ax[:,2]:
    x.set_ylabel(r'$J_z\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')

fig.tight_layout()
plt.savefig('schmactions_one_orbit.pdf')
