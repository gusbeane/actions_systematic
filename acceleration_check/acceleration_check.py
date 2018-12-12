import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import astropy.units as u

from joblib import Parallel, delayed
import multiprocessing
import sys

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

np.random.seed(162)
nproc = 40

# waiting on m12m to be transferred
# glist = ['m12i', 'm12f', 'm12m']
glist = ['m12i', 'm12f']

Rsolar = 8.2
dR = 1.0
zmax = 2.0

convert_acc = 1.0/(977.7922216731284) # for converting acceleartion

start_idx = 590
end_indx = 600
snap_idx = np.arange(start_idx, end_indx+1)

def euler_matrix(theta) :
    theta = np.array(theta)
    if len(theta)==2:
        theta = np.hstack((theta, 0))
    R_x = np.array([[1,         0,                  0  ] ,
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0 ] ,
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    # R = np.dot(R_x, np.dot(R_y, R_z))

    # this is the real one
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def euler_rotate(theta, vec):
    mat = euler_matrix(theta)
    return np.transpose(np.tensordot(mat, np.transpose(vec), axes=1))


#for gal in glist:
i = int(sys.argv[1])
gal = glist[i]
if True:
    # gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'

    sim_directory = '/mnt/ceph/users/firesims/fire2/cr_heating_fix/'+gal+'_res7100'
    snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', snap_idx,
                                        properties=['position', 'id',
                                                    'mass', 'velocity',
                                                    'form.scalefactor', 'acceleration'],
                                        simulation_directory=sim_directory,
                                        assign_principal_axes=True)

    pos = snap[-1]['star'].prop('host.distance.principal.cylindrical')
    Rbool = np.logical_and(pos[:,0] > Rsolar - dR, pos[:,0] < Rsolar + dR)
    zbool = np.abs(pos[:,1]) < zmax
    keys = np.where(np.logical_and(Rbool, zbool))[0]

    tlist = np.array([s.snapshot['time'] for s in snap])
    tlist = (tlist - tlist[0]) * 1000
    np.save('tlist_'+gal+'.npy', tlist)

    star_ids = snap[-1]['star']['id'][keys]
    star_ids = np.intersect1d(star_ids, snap[0]['star']['id'])
    star_ids = np.sort(star_ids)
    
    np.save('star_ids_'+gal+'.npy', star_ids)

    all_snap_keys = []
    for s in snap:
        this_star_ids_keys = []
        sorted_keys = s['star']['id'].argsort()
        sort = np.sort(s['star']['id'])
        j = 0
        nxt = star_ids[j]
        for sid,key in tqdm(zip(sort,sorted_keys)):
            if sid == nxt:
                this_star_ids_keys.append(key)
                j += 1
                if j == len(star_ids):
                    break
                else:
                    nxt = star_ids[j]
        all_snap_keys.append(this_star_ids_keys)
    all_snap_keys = np.array(all_snap_keys)

    all_star_pos = np.array([ s['star']['position'] for s in snap ])
    all_star_vel = np.array([ s['star']['velocity'] for s in snap ])
    all_star_acc = np.array([ s['star']['acceleration'] * convert_acc for s in snap ])

    pts_pos_list = np.array([p[k] for p,k in zip(all_star_pos, all_snap_keys) ])
    pts_vel_list = np.array([p[k] for p,k in zip(all_star_vel, all_snap_keys) ])
    pts_acc_list = np.array([p[k] for p,k in zip(all_star_acc, all_snap_keys) ])
    pts_pos_list = np.transpose(pts_pos_list, axes=(1,0,2))
    pts_vel_list = np.transpose(pts_vel_list, axes=(1,0,2))
    pts_acc_list = np.transpose(pts_acc_list, axes=(1,0,2))
    np.save('pts_pos_list_'+gal+'.npy', pts_pos_list)
    np.save('pts_vel_list_'+gal+'.npy', pts_vel_list)
    np.save('pts_acc_list_'+gal+'.npy', pts_acc_list)

    # now try to fit each pts_acc with a circle (up to Euler rotation)
    def circle(x):
        traj = np.transpose([x[0] * np.cos(x[1] * tlist), x[0] * np.sin(x[1]*tlist), np.zeros(len(tlist))]) + np.array([x[2], x[3], x[4]])
        return euler_rotate(x[5:8], traj)

    def chisq(x, pts):
        return np.sum(np.square(np.subtract(circle(x), pts)))
    
    xinit = np.zeros(8)

    def get_res(pts):
        try:
            res = minimize(chisq, xinit, args=(pts,), method='Nelder-Mead', options={'maxiter': 100000})
            return res
        except:
            return np.nan
    res_list = Parallel(n_jobs=nproc) (delayed(get_res)(pts) for pts in tqdm(pts_acc_list))
    np.save('res_list_'+gal+'.npy', res_list)


