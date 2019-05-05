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
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath} \usepackage{bm}']

textwidth = 7.10000594991
columnwidth = 3.35224200913

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

histmin = 15
histmax = 30

histmin_thin = 0
histmax_thin = 5

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

zout_thin = pickle.load(open('zout_thin.p', 'rb'))
xout_thin = pickle.load(open('xout_thin.p', 'rb'))
res_thin = pickle.load(open('true_res_thin.p', 'rb'))
J0_thin, J1_thin, J2_thin = res_thin['actions'].to_value(u.kpc*u.km/u.s)

init_pos = [8, 0, 0] * u.kpc
init_vel = [0, -190, 50] * u.km/u.s
init_vel_thin = [0, -190, 10] * u.km/u.s

s = schmactions(init_pos, init_vel)
sthin = schmactions(init_pos, init_vel_thin)

zact = s.extract_actions(zout)
xact = s.extract_actions(xout)

ztime = s.extract_time(zout)
xtime = s.extract_time(xout)

zact_thin = sthin.extract_actions(zout_thin)
ztime_thin = sthin.extract_time(zout_thin)

xact_thin = sthin.extract_actions(xout_thin)
xtime_thin = sthin.extract_time(xout_thin)

fig, ax = plt.subplots(2, 3, figsize=(textwidth, 4), sharex=True)

for x,t,a in zip(ax, (ztime, xtime), (zact, xact)):
    # get keys corresponding to 4 sigmaclip
    k0, k1, k2 = sclip(a)
    x[0].plot(t[k0], a[:,0][k0], c=tb_c[0])
    x[1].plot(t[k1], a[:,1][k1], c=tb_c[0])
    x[2].plot(t[k2], a[:,2][k2], c=tb_c[0])

    x[0].plot(t, np.full(len(t), J0), c=tb_c[0], ls='dashed')
    x[1].plot(t, np.full(len(t), J1), c=tb_c[0], ls='dashed')
    x[2].plot(t, np.full(len(t), J2), c=tb_c[0], ls='dashed')

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
    x.set_ylabel(r'$J_{R,\text{obs}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
for x in ax[:,1]:
    x.set_ylabel(r'$J_{\phi,\text{obs}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
for x in ax[:,2]:
    x.set_ylabel(r'$J_{z,\text{obs}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')

ax[0][1].set_title(r'$z\,\text{offset}=100\,\text{pc}$')
ax[1][1].set_title(r'$x\,\text{offset}=100\,\text{pc}$')

fig.tight_layout()
plt.savefig('schmactions_one_orbit.pdf')
plt.close()

# # #
# JR (x error) histogram
# # #

fig, ax = plt.subplots(2,1, figsize=(columnwidth,6.5))

k0, k1, k2 = sclip(zact)
dJRJR = 0.5*(np.percentile(xact[:,0][k0], 95) - np.percentile(xact[:,0][k0], 5))

ax[0].hist(xact[:,0][k0], bins=np.linspace(J0 - 5, J0 + 5, 60),
        edgecolor=tb_c[0], fc='none', histtype='stepfilled', linewidth=1.5)
ax[0].arrow(J0, 38, dJRJR, 0, head_width=1, head_length=0.4, length_includes_head=True, color='k')
ax[0].text(J0+dJRJR/3.5, 38.5, r'$\Delta J_R$', color='k')

ax[0].axvline(x=J0, color='k', ls='dashed', lw=1)
ax[0].set_ylabel(r'$\text{count}$')
ax[0].text(27.25, 40, r'\text{thick-disk orbit}', color='k')

k0t, k1t, k2t = sclip(xact_thin)
dJRJR_thin = 0.5*(np.percentile(xact_thin[:,0][k0t], 95) - np.percentile(xact_thin[:,0][k0t], 5))

ax[1].hist(xact_thin[:,0][k0t], bins=np.linspace(J0_thin-5, J0_thin+5, 60),
        edgecolor=tb_c[0], fc='none', histtype='stepfilled', linewidth=1.5)
ax[1].arrow(J0_thin, 30, dJRJR_thin, 0, head_width=0.7, head_length=0.35, length_includes_head=True, color='k')
ax[1].text(J0_thin+dJRJR_thin/3.5, 30.5, r'$\Delta J_R$', color='k')

ax[1].axvline(x=J0_thin, color='k', ls='dashed', lw=1)
ax[1].set_ylabel(r'$\text{count}$')
ax[1].text(35.1, 32, r'\text{thin-disk orbit}', color='k')

ax[1].set_xlabel(r'$J_{R,\text{obs}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')

fig.tight_layout()
plt.savefig('schmactions_JR_xerr_hist.pdf')

# # #
# JR (z error) histogram
# # #

fig, ax = plt.subplots(2,1, figsize=(columnwidth,6.5))

k0, k1, k2 = sclip(zact)
dJRJR = 0.5*(np.percentile(zact[:,0][k0], 95) - np.percentile(zact[:,0][k0], 5))

ax[0].hist(zact[:,0][k0], bins=np.linspace(J0 - 2, J0 + 2, 60),
        edgecolor=tb_c[0], fc='none', histtype='stepfilled', linewidth=1.5)
ax[0].arrow(J0, 35, dJRJR, 0, head_width=1, head_length=0.2, length_includes_head=True, color='k')
ax[0].text(J0+dJRJR/15, 35.5, r'$\Delta J_R$', color='k')

ax[0].axvline(x=J0, color='k', ls='dashed', lw=1)
ax[0].set_ylabel(r'$\text{count}$')
ax[0].text(30.5, 45, r'\text{thick-disk orbit}', color='k')

k0t, k1t, k2t = sclip(zact_thin)
dJRJR_thin = 0.5*(np.percentile(zact_thin[:,0][k0t], 95) - np.percentile(zact_thin[:,0][k0t], 5))

ax[1].hist(zact_thin[:,0][k0t], bins=np.linspace(J0_thin-1, J0_thin+1, 60),
        edgecolor=tb_c[0], fc='none', histtype='stepfilled', linewidth=1.5)
ax[1].arrow(J0_thin, 40, dJRJR_thin, 0, head_width=1, head_length=0.07, length_includes_head=True, color='k')
ax[1].text(J0_thin+dJRJR_thin/1.5, 41, r'$\Delta J_R$', color='k')

ax[1].axvline(x=J0_thin, color='k', ls='dashed', lw=1)
ax[1].set_ylabel(r'$\text{count}$')
ax[1].text(39.35, 59, r'\text{thin-disk orbit}', color='k')

ax[1].set_xlabel(r'$J_{R,\text{obs}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')

fig.tight_layout()
plt.savefig('schmactions_JR_zerr_hist.pdf')

# # #
# Jphi (x error) histogram
# # #

fig, ax = plt.subplots(2,1, figsize=(columnwidth,6.5))

k0, k1, k2 = sclip(xact)
dJphiJphi = 0.5*(np.percentile(xact[:,1][k1], 95) - np.percentile(xact[:,1][k1], 5))

ax[0].hist(xact[:,1][k1], bins=np.linspace(J1 - 30, J1 + 30, 60),
        edgecolor=tb_c[0], fc='none', histtype='stepfilled', linewidth=1.5)
ax[0].arrow(J1, 35, dJphiJphi, 0, head_width=1, head_length=1.3, length_includes_head=True, color='k')
ax[0].text(J1+dJphiJphi/3.5, 35.75, r'$\Delta J_{\phi}$', color='k')

ax[0].axvline(x=J1, color='k', ls='dashed', lw=1)
ax[0].set_ylabel(r'$\text{count}$')
ax[0].text(-1550, 41, r'\text{thick-disk orbit}', color='k')

k0t, k1t, k2t = sclip(xact_thin)
dJphiJphi_thin = 0.5*(np.percentile(xact_thin[:,1][k1t], 95) - np.percentile(xact_thin[:,1][k1t], 5))

ax[1].hist(xact_thin[:,1][k1t], bins=np.linspace(J1_thin-30, J1_thin+30, 60),
        edgecolor=tb_c[0], fc='none', histtype='stepfilled', linewidth=1.5)
ax[1].arrow(J1_thin, 35, dJphiJphi_thin, 0, head_width=1, head_length=1.3, length_includes_head=True, color='k')
ax[1].text(J1_thin+dJphiJphi_thin/3.5, 35.75, r'$\Delta J_{\phi}$', color='k')

ax[1].axvline(x=J1_thin, color='k', ls='dashed', lw=1)
ax[1].set_ylabel(r'$\text{count}$')
ax[1].text(-1550, 44, r'\text{thin-disk orbit}', color='k')

ax[1].set_xlabel(r'$J_{\phi,\text{obs}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')

fig.tight_layout()
plt.savefig('schmactions_Jphi_xerr_hist.pdf')

# # # 
# Jz (z error) histogram
# # #

fig, ax = plt.subplots(2,1, figsize=(columnwidth,6.5))

k0, k1, k2 = sclip(zact)
dJzJz = 0.5*(np.percentile(zact[:,2][k2], 95) - np.percentile(zact[:,2][k2], 5))

ax[0].hist(zact[:,2][k2], bins=np.linspace(histmin, histmax, 60),
        edgecolor=tb_c[0], fc='none', histtype='stepfilled', linewidth=1.5)
ax[0].arrow(J2, 60, dJzJz, 0, head_width=1, head_length=0.4, length_includes_head=True, color='k')
ax[0].text(J2+dJzJz/4, 61, r'$\Delta J_z$', color='k')

ax[0].axvline(x=J2, color='k', ls='dashed', lw=1)
ax[0].set_ylabel(r'$\text{count}$')
ax[0].text(15, 78, r'\text{thick-disk orbit}', color='k')

k0t, k1t, k2t = sclip(zact_thin)
dJzJz_thin = 0.5*(np.percentile(zact_thin[:,2][k2t], 95) - np.percentile(zact_thin[:,2][k2t], 5))

ax[1].hist(zact_thin[:,2][k2t], bins=np.linspace(histmin_thin, histmax_thin, 60),
        edgecolor=tb_c[0], fc='none', histtype='stepfilled', linewidth=1.5)
ax[1].arrow(J2_thin, 60, dJzJz_thin, 0, head_width=1, head_length=0.15, length_includes_head=True, color='k')
ax[1].text(J2_thin+dJzJz_thin/3.5, 61.5, r'$\Delta J_z$', color='k')

ax[1].axvline(x=J2_thin, color='k', ls='dashed', lw=1)
ax[1].set_ylabel(r'$\text{count}$')
ax[1].text(1, 104, r'\text{thin-disk orbit}', color='k')

ax[1].set_xlabel(r'$J_{z,\text{obs}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')


fig.tight_layout()
plt.savefig('schmactions_Jz_zerr_hist.pdf')

