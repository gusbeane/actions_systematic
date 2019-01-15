import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pykdgrav import ConstructKDTree, GetAccelParallel
from astropy.constants import G as G_astropy
from scipy.optimize import root_scalar
import astropy.units as u

import sys
import itertools
from joblib import Parallel, delayed
import multiprocessing

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

nbootstrap = 1000
np.random.seed(162)

rcut = 0.5
zcut = 1.0
nspoke = 50

Rsolar = 8.2
Rmin = 7.2
Rmax = 9.2
dR = 0.1

nproc = int(sys.argv[1])

# options for pykdgrav
Rcut = 50.0 # kpc, max R (in fid.) to use particles
theta = 0.5
G = G_astropy.to_value(u.kpc**2 * u.km / (u.s * u.Myr * u.Msun))

star_softening_in_kpc = 2.8 * 4 / 1000
dark_softening_in_kpc = 2.8 * 40 / 1000

def _setup_tree_(snap):
    star_mass = snap['star'].prop('mass')
    gas_mass = snap['gas'].prop('mass')
    dark_mass = snap['dark'].prop('mass')
    m = np.concatenate((star_mass, gas_mass, dark_mass))

    star_pos = snap['star'].prop('host.distance.principal')
    gas_pos = snap['gas'].prop('host.distance.principal')
    dark_pos = snap['dark'].prop('host.distance.principal')
    r = np.concatenate((star_pos, gas_pos, dark_pos))

    star_softening = np.full(snap['star']['mass'].shape, star_softening_in_kpc)
    dark_softening = np.full(snap['dark']['mass'].shape, star_softening_in_kpc)
    gas_softening = 2.8 * snap['gas']['smooth.length']/(1000**2) # due to bug in gizmo analysis
    soft = np.concatenate((star_softening, gas_softening, dark_softening))

    rmag = np.linalg.norm(r, axis=1)
    keys = np.where(rmag < Rcut)[0]
    r = r[keys]
    m = m[keys]
    soft = soft[keys]

    tree = ConstructKDTree( np.float64(r), np.float64(m), np.float64(soft))
    
    return tree


def read_snap(gal):
    # takes in galaxy (string = m12i, m12f, m12m, etc.)
    # reads in and sets fiducial coordinates
    # returns snap
    gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'
    sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/'+gal+'_res7100'
    snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600,
                                        properties=['position', 'id',
                                                    'mass', 'velocity',
                                                    'form.scalefactor',
                                                    'smooth.length'],
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

    tree = _setup_tree_(snap)

    return snap, tree

def gen_pos():
    theta = np.linspace(0, 2.*np.pi, nspoke)

    posx = Rsolar * np.cos(theta)
    posy = Rsolar * np.sin(theta)
    posz = np.zeros(len(posx))
    pos = np.transpose([posx, posy, posz])
    return theta, pos

def get_init_keys(p, star_pos):
    pos_diff = np.subtract(star_pos, p)
    rmag = np.linalg.norm(pos_diff[:,:2], axis=1)
    rbool = rmag < rcut
    zbool = np.abs(pos_diff[:,2]) < 2.0 * zcut
    keys = np.where(np.logical_and(rbool, zbool))[0]
    return keys

def get_keys(p, part):
    pos_diff = np.subtract(part, p)
    rmag = np.linalg.norm(pos_diff[:,:2], axis=1)
    rbool = rmag < rcut
    zbool = np.abs(pos_diff[:,2]) < zcut
    keys = np.where(np.logical_and(rbool, zbool))[0]
    return keys

def _midplane_med_(pos, init_pos, init_vel):
    mid_pos = pos.copy()
    for _ in range(10):
        keys = get_keys(mid_pos, init_pos)
        mid_pos[2] = np.median(init_pos[:,2][keys])
    mid_vel = np.median(init_vel[:,2][keys])
    return mid_pos[2], mid_vel

class _midplane_force_(object):
    def __init__(self, tree):
        self.tree = tree

    def _force_z_(self, z, xy):
        r = np.append(xy, z)
        r = np.reshape(r, (1, 3))
        accel = GetAccelParallel(r, self.tree, G, theta)
        return accel[0][2]

    def __call__(self, pos, init_pos, init_vel):
        zinit = pos[2]
        xy = pos[0:2]
        res = root_scalar(self._force_z_, args=(xy,), x0=zinit, bracket=(-1, 1))
        return res.root, 0

    

def get_midplane_with_error(pos, star_pos, star_vel, force=False, tree=None):
    if force:
        if tree is None:
            print('pass tree when using force!')
            sys.exit(1)
        midplane = _midplane_force_(tree)
    else:
        midplane = _midplane_med_

    # get all particles within 2x zheight
    init_keys = get_init_keys(pos, star_pos)
    init_pos = star_pos[init_keys]
    init_vel = star_vel[init_keys]

    # calculate midplane using all particles
    midplane_central, midplane_vel = midplane(pos, init_pos, init_vel)
    
    # prepare to bootstrap
    if force:
        l = 0.9*midplane_central
        h = 1.1*midplane_central
        l_v = 0.9*midplane_vel
        h_v = 1.1*midplane_vel
    else:
        keys_to_choose = list(range(len(init_pos)))
        rand_choice = np.random.choice(keys_to_choose, len(init_pos)*nbootstrap)
        rand_choice = np.reshape(rand_choice, (nbootstrap, len(init_pos)))
        init_pos_rand = init_pos[rand_choice]
        init_vel_rand = init_vel[rand_choice]
        med_rand = np.array([ midplane(pos, ipos, ivel) for ipos,ivel in zip(init_pos_rand,init_vel_rand) ])
        dist_pos = np.subtract(med_rand[:,0], midplane_central)
        dist_vel = np.subtract(med_rand[:,1], midplane_vel)
        up_pos = np.percentile(dist_pos, 95)
        low_pos = np.percentile(dist_pos, 5)
        up_vel = np.percentile(dist_vel, 95)
        low_vel = np.percentile(dist_vel, 5)
        l = midplane_central - up_pos
        h = midplane_central - low_pos
        l_v = midplane_vel - up_vel
        h_v = midplane_vel - low_vel
    return midplane_central, l, h, midplane_vel, l_v, h_v

def fit(x, theta):
    return x[0]*np.cos(theta+x[1]) + x[2]

def chisq(x, theta, midplane_est):
    return np.sum(np.square(np.subtract(fit(x, theta), midplane_est)))

def main(gal):
    snap, tree = read_snap(gal)
    
    star_pos = snap['star'].prop('host.distance.principal')
    star_vel = snap['star'].prop('host.velocity.principal')
    
    theta, pos = gen_pos()

    force=True

    if force:
        result = np.array([ get_midplane_with_error(p, star_pos, star_vel, force=force, tree=tree) for p in tqdm(pos) ])
    else:
        result = Parallel(n_jobs=nproc) (delayed(get_midplane_with_error)(p, star_pos, star_vel, force=force, tree=tree) for p in tqdm(pos))
        result = np.array(result)

    midplane_est = result[:,0]

    res = minimize(chisq, np.array([0.1, 0, 0]), args=(theta, midplane_est), method='Nelder-Mead')
    A = res.x[0]
    B = res.x[1]
    C = res.x[2]
    print('A=', A, 'B=', B, 'C=', C)
    fit = A*np.cos(theta + B) + C

    out = np.concatenate((theta.reshape(nspoke, 1), result, fit.reshape(nspoke,1)), axis=1)

    if force:
        np.save('output/out_force_'+gal+'.npy', out)
    else:
        np.save('output/out_'+gal+'.npy', out)

if __name__ == '__main__':
    glist = ['m12i', 'm12f', 'm12m']
    for gal in glist:
        main(gal)

