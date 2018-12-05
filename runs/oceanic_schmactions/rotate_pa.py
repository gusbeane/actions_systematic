import gizmo_analysis as gizmo
import numpy as np
import agama
import sys
import utilities as ut
from scipy.optimize import minimize

np.random.seed(162)
agama.setUnits(mass=1, length=1, velocity=1)

snap_idx=600
sim_dir = '/mnt/ceph/users/firesims/fire2/metaldiff/m12i_res7100/'
gal_info = 'm12i_info.txt'

cadence_list = [50, 20]
#cadence_list = [50]

tend = 2000 # Myr
dt = 0.1 # Myr
tlist = np.arange(0 , tend, dt)

snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                     'index', snap_idx,
                                     simulation_directory=sim_dir,
                                     assign_center=False)

ref = np.genfromtxt(gal_info, comments='#', delimiter=',')
snap.center_position = ref[0]
snap.center_velocity = ref[1]
snap.principal_axes_vectors = ref[2:]
for k in snap.keys():
    for attr in ['center_position', 'center_velocity', 
                 'principal_axes_vectors']:
        setattr(snap[k], attr, getattr(snap, attr))

fiducial_center = ref[0]
fiducial_velocity = ref[1]
fiducial_pa = ref[2:]

def get_position(snap, pos, center, pa=None):
    values = ut.coordinate.get_distances(
                        pos, center,
                        snap.info['box.length'], snap.snapshot['scalefactor'])
    if pa is not None:
        values = ut.coordinate.get_coordinates_rotated(
                    values, pa)
    return values

def get_velocity(snap, pos, vel, center, center_velocity, pa=None):
    values = ut.coordinate.get_velocity_differences(
                        vel, center_velocity,
                        pos, center,
                        snap.info['box.length'], snap.snapshot['scalefactor'],
                        snap.snapshot['time.hubble'])
    if pa is not None:
        values = ut.coordinate.get_coordinates_rotated(
                    values, pa)
    return values

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

potential = agama.Potential(file='fiducial_pot')
af = agama.ActionFinder(potential, interp=False)

# read in simulated positions and velocities
posfile = 'pos_' + sys.argv[1] + '.npy'
velfile = 'vel_' + sys.argv[1] + '.npy'
pos_full = np.load(posfile)
vel_full = np.load(velfile)

global nframes, npart, ndim, pos, vel

def init_minimizer(cadence):
    pos = pos_full[::cadence]
    vel = vel_full[::cadence]

    nframes, npart, ndim = np.shape(pos)
    return pos, vel, nframes, npart, ndim

def actions(pos, vel):
    flat_pos = np.reshape(pos, (nframes*npart, ndim))
    flat_vel = np.reshape(vel, (nframes*npart, ndim))
    phase = np.c_[flat_pos, flat_vel]
    act = af(phase)
    return np.reshape(act, (nframes, npart, ndim))

def chisq(x, return_actions=False):
    offset = x[:3]
    theta = x[3:]

    rotated_pa = euler_rotate(theta, fiducial_pa)

    new_pos = get_position(snap, pos, offset, rotated_pa)
    new_vel = get_position(snap, vel, np.array([0,0,0]), rotated_pa)

    act = actions(new_pos, new_vel)
    act_collapse = np.median(act, axis=1)
    act_med = np.median(act_collapse, axis=0)
    if return_actions:
        return act_collapse
    act_collapse = np.subtract(act_collapse, act_med)
    return np.sum(np.square(act_collapse/act_med)[:,2])

xinit = np.array([0, 0, 0, 0, 0, 0])
for cd in cadence_list:
    pos, vel, nframes, npart, ndim = init_minimizer(cd)
    res = minimize(chisq, xinit, method='Nelder-Mead')
    xinit = res.x

offset = xinit[:3]
theta = xinit[3:]

oa_center_position = fiducial_center + offset
oa_pa = euler_rotate(theta, fiducial_pa)
oa_center_velocity = ref[1] # don't recalculate center velocity
# remember that center velocity is calculated *before* rotation

new_frame = np.vstack((oa_center_position, oa_center_velocity, oa_pa))
np.savetxt('oa_frame.txt', new_frame)

# now dump actions
fiducial_actions = chisq(np.array([0, 0, 0, 0, 0, 0]), return_actions=True) 
oa_actions = chisq(res.x, return_actions=True)

np.save('fiducial_actions_'+sys.argv[1]+'.npy', fiducial_actions)
np.save('oa_actions_'+sys.argv[1]+'.npy', oa_actions)

