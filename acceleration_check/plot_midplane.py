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


all_snap = {}
res_list = {}
star_ids = {}
pts_acc_list = {}
for gal in ['m12i', 'm12f']:
    sim_directory = '/mnt/ceph/users/firesims/fire2/cr_heating_fix/'+gal+'_res7100'
    res_list[gal] = np.load('res_list_'+gal+'.npy')
    star_ids[gal] = np.load('star_ids_'+gal+'.npy')
    pts_acc_list[gal] = np.load('pts_acc_list_'+gal+'.npy')
    
    all_snap[gal] = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', snap_idx,
                                        properties=['position', 'id',
                                                    'mass', 'velocity',
                                                    'form.scalefactor', 'acceleration'],
                                        simulation_directory=sim_directory,
                                        assign_principal_axes=True)

fig, axes = plt.subplots(2, 1, figsize=(6,6), sharex=True)
perc = 5

def get_keys_from_cylindrical(pos, Rsolar=8.2, dR=1.0, zmax=2.0):
    Rbool = np.logical_and(pos[:,0] > Rsolar - dR, pos[:,0] < Rsolar + dR)
    zbool = np.abs(pos[:,1]) < zmax
    return np.where(np.logical_and(Rbool, zbool))[0]


for gal,ax in zip(['m12i', 'm12f'], axes):
    snap = all_snap[gal]
    
    gas_cyl_pos = snap['gas'].prop('host.distance.principal.cylindrical')
    gas_mass = snap['gas']['mass']
    gas_keys = get_keys_from_cylindrical(gas_cyl_pos)
    gas_cyl_pos = gas_cyl_pos[gas_keys]
    gas_mass = gas_mass[gas_keys]

    ax.hist2d(gas_cyl_pos[:,2]/np.pi, gas_cyl_pos[:,1]*1000.0, weights=gas_mass, bins=200)


    nsample = len(pts_acc_list[gal][0])

    fun_list = np.array([ res.fun for res in res_list[gal] ])
    rms_list = np.sqrt(fun_list / nsample)
    rms_bool = rms_list < np.percentile(rms_list, perc)
    
    x_list = np.array([ res.x for res in res_list[gal] ])
    R_list = np.array([ x_list[0][0]/(x[2]**2) for x in x_list ])
    new_bool = np.logical_and(R_list > Rsolar - 2.0*dR, R_list < Rsolar + 2.0*dR)
    keys = np.where(np.logical_and(new_bool, rms_bool))
    

    circ_star_ids = star_ids[gal][keys]
    snap_keys = np.where(np.isin(snap['star']['id'], circ_star_ids))[0]

    cyl_pos = snap['star'].prop('host.distance.principal.cylindrical')
    circ_cyl_pos = cyl_pos[snap_keys]
    
    ax.scatter(circ_cyl_pos[:,2]/np.pi, circ_cyl_pos[:,1]*1000.0, c='k', s=0.2, alpha=0.5)
    ax.set_ylabel(r'$z\,[\,\text{pc}\,]$')
    ax.set_title(r'$\text{'+gal+r'}$')


axes[1].set_xlabel('$\phi/\pi$')
fig.savefig('midplane_circ_stars.pdf')

Rsolar = 8.2
dR = 1.0
zmax = 2.0

fig, axes = plt.subplots(2, 1, figsize=(6,6), sharex=True)

for gal, ax in zip(['m12i', 'm12f'], axes):
    cyl_pos = snap['star'].prop('host.distance.principal.cylindrical')
    fun_list = np.array([ res.fun for res in res_list[gal] ])
    num = int( (perc/100.0) * len(fun_list) )

    Rbool = np.logical_and(cyl_pos[:,0] > Rsolar - dR, cyl_pos[:,0] < Rsolar + dR)
    zbool = np.abs(cyl_pos[:,1]) < zmax
    keys = np.where(np.logical_and(Rbool, zbool))[0]
    rand_keys = np.random.choice(keys, num)

    rand_cyl_pos = cyl_pos[rand_keys]

    ax.scatter(rand_cyl_pos[:,2]/np.pi, rand_cyl_pos[:,1]*1000.0, c='k', s=0.2, alpha=0.5)
    ax.set_ylabel(r'$z\,[\,\text{pc}\,]$')
    ax.set_title(r'$\text{'+gal+r'}$')

axes[1].set_xlabel('$\phi/\pi$')
fig.savefig('midplane_rand_stars.pdf')

