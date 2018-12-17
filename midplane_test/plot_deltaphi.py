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

def rotate(l, n):
    return np.append(l[n:], l[:n])

def get_range_vs_dphi(theta, midplane):
    dphi_list_list = []
    r_list_list = []
    for i in range(len(theta)):
        m = rotate(midplane, i)
        t = rotate(theta, i)
        t -= t[0]
        t = np.mod(t + np.pi, 2.*np.pi) - np.pi
        dphi_list = [0]
        r_list = [0]
        for j in rotate(range(int(len(theta)/2.0)), 1):
            p = np.append(t[-j:], t[0:j+1]) 
            rl = np.append(m[-j:], m[0:j+1])
            dphi = np.max(p) - np.min(p)
            r = np.max(rl) - np.min(rl)
            dphi_list.append(dphi)
            r_list.append(r)
        dphi_list_list.append(dphi_list)
        r_list_list.append(r_list)
    return np.array(dphi_list_list), np.array(r_list_list)


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8,3.5/2))
# now make paper plot, with just fit
for gal,ax_col in zip(glist, ax.transpose()):
    midplane_est = np.load('midplane_est_'+gal+'.npy')
    err_low = np.load('err_low_'+gal+'.npy')
    err_high = np.load('err_high_'+gal+'.npy')

    ax_col.set_xlabel(r'$\Delta \phi/\pi$')

    ax_col.text(0.05, 0.88, gal, 
               horizontalalignment='left', 
               verticalalignment='center', 
               transform = ax_col.transAxes)

    ax_col.set_xlim(0, 2)

    ax_col.set_ylim(0, 400)

    fit = np.load('fit_'+gal+'.npy')

    m = midplane_est - fit

    dphi_list_list, r_list_list = get_range_vs_dphi(theta, m)
    for dphi_list, r_list in zip(dphi_list_list, r_list_list):
        ax_col.plot(dphi_list/np.pi, r_list*1000,  c=tb_c[-1], alpha=0.15, lw=1)

    dphi_mean = np.mean(dphi_list_list, axis=0) 
    r_mean = np.mean(r_list_list, axis=0) 
    ax_col.plot(dphi_mean/np.pi, r_mean*1000, c=tb_c[0])

ax[0].set_ylabel(r'$\text{range}\,[\,\text{pc}\,]$')

fig.tight_layout()
fig.savefig('range_dphi.pdf')
