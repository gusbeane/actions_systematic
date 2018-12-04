import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

tend = 2000 # Myr
dt = 0.1 # Myr
tlist = np.arange(0 , tend, dt)

cadence = 20
tlist = tlist[::cadence]

tmax = 1000

#idx_list = ['0', '1']
idx_list = ['0']
for idx in idx_list:
    fig, ax = plt.subplots(1, 3, figsize=(7, 2.25))
    fiducial_actions = np.load('fiducial_actions_'+idx+'.npy')
    oa_actions = np.load('oa_actions_'+idx+'.npy')
    
    ax[0].plot(tlist, fiducial_actions[:,0], c=tb_c[-1], ls='dashed')
    ax[1].plot(tlist, fiducial_actions[:,2], c=tb_c[-1], ls='dashed')
    ax[2].plot(tlist, fiducial_actions[:,1], c=tb_c[-1], ls='dashed')
    ax[0].plot(tlist, oa_actions[:,0], c=tb_c[0])
    ax[1].plot(tlist, oa_actions[:,2], c=tb_c[0])
    ax[2].plot(tlist, oa_actions[:,1], c=tb_c[0])
    
    for x in ax:
        x.set_xlim(0, tmax)
        x.set_xlabel(r'$t\,[\,\text{Myr}\,]$')
    
    ax[0].set_ylabel(r'$J_r\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
    ax[1].set_ylabel(r'$L_z\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
    ax[2].set_ylabel(r'$J_z\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')

    fig.tight_layout()
    fig.savefig('actions_time_'+idx+'.pdf')

