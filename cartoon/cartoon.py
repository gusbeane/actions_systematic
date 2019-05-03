import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = {'blue': '#4e79a7',
        'orange': '#f28e2b', 
        'red': '#e15759', 
        'cyan': '#76b7b2', 
        'green': '#59a14f',
        'yellow': '#edc948', 
        'purple': '#b07aa1', 
        'pink': '#ff9da7', 
        'brown': '#9c755f', 
        'gray': '#bab0ac'}

textwidth = 7.10000594991
columnwidth = 3.35224200913

fig, ax = plt.subplots(1, figsize=(textwidth, 3))

x = np.linspace(0, 7*np.pi, 5000)
y = np.cos(x)
y2 = 0.25*np.cos(x + np.pi) - 1.25
y3 = 2.25*np.cos(x) - 1.25

small_color = tb_c['cyan']


ax.plot(x, y, c=tb_c['gray'])

ax.plot(x, y2, c=tb_c['orange'], zorder=4)
ax.scatter(3.*np.pi, -1, c=tb_c['orange'], zorder=4)

ax.plot(x, y3, c=tb_c['cyan'], zorder=5)
ax.scatter(4.*np.pi, 1, c=tb_c['cyan'], zorder=5)

ax.plot(x, np.full(len(x), 0), c=tb_c['gray'])
ax.plot(x, np.full(len(x), -1.25), c=tb_c['gray'], ls='dashed')
#ax.axis('off')

ax.set_xlabel(r'$\text{orbital angle}$')
ax.set_ylabel(r'$z$')

ax.set_xticklabels([])
ax.set_yticklabels([])

fig.tight_layout()
fig.savefig('cartoon.pdf', bbox_inches='tight', pad_inches=0)
