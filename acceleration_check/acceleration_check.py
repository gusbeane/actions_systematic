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

np.random.seed(162)

# waiting on m12m to be transferred
# glist = ['m12i', 'm12f', 'm12m']
glist = ['m12i', 'm12f']

for gal in glist:
    # gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'

    sim_directory = '/mnt/ceph/users/firesims/fire2/cr_heating_fix/'+gal+'_res7100'
    snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600,
                                        properties=['position', 'id',
                                                    'mass', 'velocity',
                                                    'form.scalefactor', 'acceleration'],
                                        simulation_directory=sim_directory)

