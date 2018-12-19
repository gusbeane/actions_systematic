import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

nbootstrap = 1000
np.random.seed(162)

rcut = 0.5
zcut = 1.0
nspoke = 50


glist = ['m12i', 'm12f', 'm12m']

theta = np.load('theta.npy')

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8,3.5))
# first make midplane comparison plot
for gal,ax_col in zip(glist, ax.transpose()):
    midplane_est = np.load('midplane_est_'+gal+'.npy')
    err_low = np.load('err_low_'+gal+'.npy')
    err_high = np.load('err_high_'+gal+'.npy')

    ax_col[0].plot(theta/np.pi, midplane_est*1000, c=tb_c[0])
    ax_col[0].plot(theta/np.pi, err_low*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[0].plot(theta/np.pi, err_high*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[0].fill_between(theta/np.pi, err_high*1000, err_low*1000, color=tb_c[0], alpha=0.25)

    ax_col[1].set_xlabel(r'$\phi/\pi$')

    ax_col[0].text(0.05, 0.88, gal, 
               horizontalalignment='left', 
               verticalalignment='center', 
               transform = ax_col[0].transAxes)

    ax_col[0].set_xlim(0, 2)
    ax_col[1].set_xlim(0, 2)

    ax_col[0].set_ylim(-200, 200)
    ax_col[1].set_ylim(-200, 200)

    fit = np.load('fit_'+gal+'.npy')

    ax_col[1].plot(theta/np.pi, (midplane_est-fit)*1000, c=tb_c[0])
    ax_col[1].plot(theta/np.pi, (err_low-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[1].plot(theta/np.pi, (err_high-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[1].fill_between(theta/np.pi, (err_high-fit)*1000, (err_low-fit)*1000, color=tb_c[0], alpha=0.25)

ax[0][0].set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')
ax[1][0].set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')

fig.tight_layout()
plt.savefig('midplane.pdf')





fig, ax = plt.subplots(2, 3, sharex=True, figsize=(8,3.5))
# now make paper plot, with just fit
for gal,ax_col in zip(glist, ax.transpose()):
    midplane_est = np.load('midplane_est_'+gal+'.npy')
    err_low = np.load('err_low_'+gal+'.npy')
    err_high = np.load('err_high_'+gal+'.npy')

    midplane_vel = np.load('midplane_vel_'+gal+'.npy')
    err_vel_low = np.load('err_vel_low_'+gal+'.npy')
    err_vel_high = np.load('err_vel_high_'+gal+'.npy')

    ax_col[0].set_xlabel(r'$\phi/\pi$')

    ax_col[0].text(0.05, 0.88, gal, 
               horizontalalignment='left', 
               verticalalignment='center', 
               transform = ax_col[0].transAxes)

    ax_col[0].set_xlim(0, 2)
    ax_col[1].set_xlim(0, 2)

    ax_col[0].set_ylim(-200, 200)

    fit = np.load('fit_'+gal+'.npy')

    ax_col[0].plot(theta/np.pi, (midplane_est-fit)*1000, c=tb_c[0])
    ax_col[0].plot(theta/np.pi, (err_low-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[0].plot(theta/np.pi, (err_high-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[0].fill_between(theta/np.pi, (err_high-fit)*1000, (err_low-fit)*1000, color=tb_c[0], alpha=0.25)
    
    ax_col[1].plot(theta/np.pi, (midplane_vel), c=tb_c[1])
    ax_col[1].plot(theta/np.pi, (err_vel_low), c=tb_c[1], ls='dashed', alpha=0.5)
    ax_col[1].plot(theta/np.pi, (err_vel_high), c=tb_c[1], ls='dashed', alpha=0.5)
    ax_col[1].fill_between(theta/np.pi, (err_vel_high), (err_vel_low), color=tb_c[1], alpha=0.25)

ax[0][0].set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')

fig.tight_layout()
plt.savefig('midplane_fit.pdf')
