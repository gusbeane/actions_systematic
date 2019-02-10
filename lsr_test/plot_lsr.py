import numpy as np 
import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

gal_list = ['m12i', 'm12f', 'm12m']

fig, ax = plt.subplots(3, 3, sharex=True, sharey='row')

for gal, x in zip(gal_list, ax.transpose()):
    fname = 'output/lsr_'+gal+'.npy'
    lsr = np.load(fname)
    x[0].plot(lsr[:,0]/np.pi, lsr[:,1], c=tb_c[0])
    x[1].plot(lsr[:,0]/np.pi, lsr[:,2], c=tb_c[0])
    x[2].plot(lsr[:,0]/np.pi, lsr[:,3], c=tb_c[0])
ax[0][0].set_ylabel(r'$v_{R,\text{LSR}}\,[\,\text{km}/\text{s}\,]$')
ax[0][0].set_ylim(-100, 100)

ax[1][0].set_ylabel(r'$v_{z,\text{LSR}}\,[\,\text{km}/\text{s}\,]$')
ax[1][0].set_ylim(-50, 50)

ax[2][0].set_ylabel(r'$v_{\phi,\text{LSR}}\,[\,\text{km}/\text{s}\,]$')
ax[2][0].set_ylim(-300, -150)

for x in ax[2]:
    x.set_xlabel(r'$\phi/\pi$')
    x.set_xlim(0, 2)

fig.tight_layout()
fig.savefig('lsr.pdf')