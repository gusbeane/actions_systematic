import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

def center(x):
    hlx = x[:4]
    x0 = x[4:7]
    theta = x[7:10]
    out = np.transpose([np.zeros(len(tlist)), np.zeros(len(tlist)), hlx[2]*tlist])
    out += x0
    out = euler_rotate(theta, out)
    return out


pts_list = {}
res_list = {}
for gal in ['m12i', 'm12f', 'm12m']:
    res_list[gal] = np.load('res_list_'+gal+'.npy')
    pts_list[gal] = np.load('pts_list_'+gal+'.npy')

# from each res.x, we can compute where that "circular" orbit
# thinks the center of the galaxy is, and where the midplane is
# can only trust stars with orbits close enough to circular...
# first, let's just peek at the rms errors of each fit

def plot_hist(gal, rms_list, out, perc_cut=10, bins=None):
    fig, ax = plt.subplots(1, 1, figsize=(4,3))

    if bins is None:
        bin_min = np.floor(np.min(rms_list))
        bin_max = np.ceil(np.max(rms_list))
        bins = np.linspace(bin_min, bin_max, 1.0)

    plt.hist(rms_list, bins, c=tb_c[0])
    ax.set_xlabel(r'$\text{rms}\,[\,\text{pc}\,]$')
    ax.set_ylabel(r'$\text{count}$')
    ax.set_title(r'$\text{'+gal+r'}$')

    rms_perc = np.percentile(rms_list, perc_cut)
    #ax.axvline(rms_perc, color=tb_c[-1])

    fig.tight_layout()
    fig.savefig(out)

fun_list = {}
rms_list = {}
for gal in res_list.keys():
    fun_list[gal] = np.array([ res.fun for res in res_list[gal] ])
    rms_list[gal] = 1000.0 * np.sqrt(fun_list[gal]/11)
    plot_hist(gal, rms_list[gal], 'rms_counts_'+gal+'.pdf') 
