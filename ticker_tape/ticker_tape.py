import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

#gal_info = 'oa_frame.txt'
gal_list = ['m12i', 'm12f', 'm12m']

for gal in gal_list:
    gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'

    sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/' + gal + '_res7100'
    snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600,
                                   properties=['position', 'id',
                                               'mass', 'velocity',
                                               'form.scalefactor'],
                                     assign_center=False,
                                     simulation_directory=sim_directory)

    ref = np.genfromtxt(gal_info, comments='#', delimiter=',')
    snap.center_position = ref[0]
    snap.center_velocity = ref[1]
    snap.principal_axes_vectors = ref[2:]
    for k in snap.keys():
        for attr in ['center_position', 'center_velocity', 
                     'principal_axes_vectors']:
            setattr(snap[k], attr, getattr(snap, attr))

    star_pos = snap['star'].prop('host.distance.principal.cylindrical')

    R = star_pos[:,0]
    z = star_pos[:,1]
    phi = star_pos[:,2]

    Rmid = 8.2
    dR = 0.5

    Rbool = np.logical_and(R > Rmid - dR, R < Rmid + dR)
    zbool = np.abs(z) < 1

    keys = np.where(np.logical_and(Rbool, zbool))[0]

    phibins = np.linspace(0, 2.*np.pi, 51)
    zbins = np.arange(-1, 1, 0.01)
    plt.figure(figsize=(7,2))
    plt.hist2d(phi[keys], z[keys], bins=(phibins, zbins))

    plt.savefig('ticker_tape_'+gal+'.pdf')
    plt.close()

