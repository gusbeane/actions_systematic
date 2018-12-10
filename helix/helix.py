import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

nbootstrap = 1000
np.random.seed(162)

Rsolar = 8.2
dR = 1.0

nproc = 40

glist = ['m12i', 'm12f', 'm12m']

idx_start = 590
idx_end = 600
snap_idx = np.arange(idx_start, idx_end+1)

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

def helix(x):
    hlx = x[:4]
    x0 = x[4:7]
    theta = x[7:10]
    acc = x[10:13]
    out = np.transpose([hlx[0]*np.cos(hlx[3]*tlist), hlx[1]*np.sin(hlx[3]*tlist), hlx[2]*tlist])
    out += x0 + np.multiply(0.5, np.transpose(np.outer(acc, np.square(tlist))))
    out = euler_rotate(theta, out)
    return out

def chisq(x, pts):
    guess = helix(x)
    return np.sum(np.square(np.subtract(pts, guess)))


#for gal,ax_col in zip(glist, ax.transpose()):
#for gal in glist:

i = int(sys.argv[1])
gal = glist[i]

if True:
    gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'

    sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/'+gal+'_res7100'
    snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', snap_idx,
                                        properties=['position', 'id',
                                                    'mass', 'velocity',
                                                    'form.scalefactor'],
                                        assign_center=False,
                                        simulation_directory=sim_directory)

    #tlist = np.array([
    tlist = np.array([s.snapshot['time'] for s in snap])
    tlist = (tlist - tlist[0]) * 1000
    np.save('tlist_'+gal+'.npy', tlist)

    ref = np.genfromtxt(gal_info, comments='#', delimiter=',')
    snap[-1].center_position = ref[0]
    snap[-1].center_velocity = ref[1]
    snap[-1].principal_axes_vectors = ref[2:]
    for k in snap[-1].keys():
        for attr in ['center_position', 'center_velocity', 
                     'principal_axes_vectors']:
            setattr(snap[-1][k], attr, getattr(snap[-1], attr))
    

    star_pos_z0 = snap[-1]['star'].prop('host.distance.principal.cylindrical')
    
    Rbool = np.logical_and(star_pos_z0[:,0] > Rsolar - dR, star_pos_z0[:,0] < Rsolar + dR)
    keys = np.where(Rbool)[0]

    star_ids = snap[-1]['star']['id'][keys]
    star_ids = np.intersect1d(star_ids, snap[0]['star']['id'])

    star_ids = np.sort(star_ids)
    
    #all_snap_keys = np.array([ np.intersect1d(star_ids, s['star']['id'], assume_unique=True, return_indices=True)[2].tolist() for s in snap ])
    
    xinit = np.zeros(13)
    xinit[3:6] = snap[-1].center_position
   
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

    pts_list = np.array([p[k] for p,k in zip(all_star_pos, all_snap_keys) ])
    
    pts_list = np.transpose(pts_list, axes=(1,0,2))
    np.save('pts_list_'+gal+'.npy', pts_list)

    def get_res(pts):
        try:
            res = minimize(chisq, xinit, args=(pts,), method='Nelder-Mead', options={'maxiter': 100000})
            return res    
        except:
            return np.nan
    
    res_list = Parallel(n_jobs=nproc) (delayed(get_res)(pts) for pts in tqdm(pts_list))
    np.save('res_list_'+gal+'.npy', res_list)
