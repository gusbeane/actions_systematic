import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from oceanic.analysis import snapshot_action_calculator
from oceanic.options import options_reader

import dill
import utilities as ut

from scipy.optimize import minimize

gal_info = np.genfromtxt('../m12i_info.txt', comments='#', delimiter=',')

cen = gal_info[0]
vel = gal_info[1]
pa = gal_info[2:]

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
    R = np.dot(R_x, np.dot(R_y, R_z))
    
    # this is the real one
    # R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def euler_rotate(theta, vec):
    mat = euler_matrix(theta)
    return np.transpose(np.tensordot(mat, np.transpose(vec), axes=1))

opt = options_reader('../options_m12i')
ac = snapshot_action_calculator(opt, snapshot_file='cluster_snapshots_m12i.p')

cluster = dill.load(open('cluster_snapshots_m12i.p', 'rb'))
npart = len(cluster[0]['position'])
nsnap = len(cluster)

pos = np.array([cl['position'] for cl in cluster])
vel = np.array([cl['velocity'] for cl in cluster])
time = np.array([cl['time'] for cl in cluster])

pos_flat = np.reshape(pos, (npart*nsnap, 3))
vel_flat = np.reshape(vel, (npart*nsnap, 3))
actions = ac.all_actions(pos=pos_flat, vel=vel_flat)

def chisq(theta, return_actions=False):
    real_theta = theta[3:]
    offset = theta[:3]
    
    pos_flat = np.reshape(pos, (npart*nsnap, 3))
    vel_flat = np.reshape(vel, (npart*nsnap, 3))
    pos_flat = np.subtract(pos_flat, offset)
    pos_flat = euler_rotate(real_theta, pos_flat)
    vel_flat = euler_rotate(real_theta, vel_flat)
    actions = ac.all_actions(pos=pos_flat, vel=vel_flat, update=False)
    actions = np.reshape(actions, (nsnap, npart, 3))
    
    act_med = np.median(actions, axis=1)
    mean_act = np.mean(act_med, axis=0)
    diff = np.subtract(act_med, mean_act)
    err = np.sum(np.square(diff))
    if return_actions:
        return actions
    else:
        return err

res = minimize(chisq, [0, 0, 0, 0, 0, 0])
