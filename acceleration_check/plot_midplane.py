import numpy as np
import gizmo_analysis as gizmo

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

snap_idx = 600

fig, axes = plt.subplots(2, 1, figsize=(6,6), sharex=True)

perc = 5

all_snap = {}
for gal in ['m12i', 'm12f']:
    sim_directory = '/mnt/ceph/users/firesims/fire2/cr_heating_fix/'+gal+'_res7100'
    all_snap[gal] = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', snap_idx,
                                        properties=['position', 'id',
                                                    'mass', 'velocity',
                                                    'form.scalefactor', 'acceleration'],
                                        simulation_directory=sim_directory,
                                        assign_principal_axes=True)

for gal,ax in zip(['m12i', 'm12f'], axes):
    snap = all_snap[gal]
    
    res_list = np.load('res_list_'+gal+'.npy')
    
    
    star_ids = np.load('star_ids_'+gal+'.npy')

    res_list = np.load('res_list_'+gal+'.npy')
    pts_acc_list = np.load('pts_acc_list_'+gal+'.npy')
    nsample = len(pts_acc_list[0])

    fun_list = np.array([ res.fun for res in res_list ])
    rms_list = np.sqrt(fun_list / nsample)
    keys = np.where(rms_list < np.percentile(rms_list, perc))[0]

    circ_star_ids = star_ids[keys]
    snap_keys = np.where(np.isin(snap['star']['id'], circ_star_ids))[0]

    cyl_pos = snap['star'].prop('host.distance.principal.cylindrical')
    circ_cyl_pos = cyl_pos[snap_keys]
    
    ax.scatter(cy

