import gizmo_analysis as gizmo
import numpy as np 
import agama
import utilities as ut
import sys
from pykdgrav import ConstructKDTree, GetAccelParallel
from amuse.units import units

from astropy.constants import G as G_astropy
import astropy.units as u

np.random.seed(162)
agama.setUnits(mass=1, length=1, velocity=1)

snap_idx=600
sim_dir = '/mnt/ceph/users/firesims/fire2/metaldiff/m12i_res7100/'
gal_info = 'm12i_info.txt'

star_char_mass = 0.048
dark_softening_in_pc = 112.0
Rmax = 50.0

ss_ids = [23693026,
          17012804]
ss_id = ss_ids[int(sys.argv[1])]

G = G_astropy.to_value(u.kpc**2 * u.km / (u.s * u.Myr * u.Msun))
theta = 0.5


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

class pykdgrav(object):
    def __init__(self, r, m, soft, theta, G):
        print('constructing kdtree')
        self.tree = ConstructKDTree( np.float64(r), np.float64(m), np.float64(soft))
        self.theta = theta
        self.G = G
        print('done')
    
    def evolve_model(self, time, timestep=None):
        pass

    def get_gravity_at_point(self, eps, xlist, ylist, zlist):
        xlist_kpc = xlist.value_in(units.kpc)
        ylist_kpc = ylist.value_in(units.kpc)
        zlist_kpc = zlist.value_in(units.kpc)
        pos = np.transpose([xlist_kpc, ylist_kpc, zlist_kpc])
        accel = GetAccelParallel(pos, self.tree, self.G, self.theta)
        ax = accel[:,0] | units.kms/units.Myr
        ay = accel[:,1] | units.kms/units.Myr
        az = accel[:,2] | units.kms/units.Myr
        return ax, ay, az


# read in snapshot
snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                     'index', snap_idx,
                                     simulation_directory=sim_dir,
                                     assign_center=False)

# set up fiducial frame
# maybe not necessary?
ref = np.genfromtxt(gal_info, comments='#', delimiter=',')
snap.center_position = ref[0]
snap.center_velocity = ref[1]
snap.principal_axes_vectors = ref[2:]
for k in snap.keys():
    for attr in ['center_position', 'center_velocity', 
                 'principal_axes_vectors']:
        setattr(snap[k], attr, getattr(snap, attr))

# # # # # 
# now set up softening
# # # # #

# we don't want to use the ss in the gravity calc (not really important)
ss_key = np.where(snap['star']['id'] != ss_id)[0]


star_mass = snap['star']['mass'][ss_key]
star_softening = np.power(star_mass/star_char_mass, 1.0/3.0)
star_softening /= 1000.0

dark_softening = np.full(len(snap['dark']['position']),
    float(dark_softening_in_pc)/1000.0)

# bug in gizmo_analysis requires dividing by 1000**2
gas_softening = 2.8 * snap['gas']['smooth.length'] / 1000.0**2

all_softening = np.concatenate((star_softening, dark_softening,
                                        gas_softening))

# set up other things necessary for tree calc
# we want to do this in the original sim frame
# except centered on the galaxy
# to reduce finite difference errors

all_position = np.concatenate((snap['star'].prop('host.distance')[ss_key],
                snap['dark'].prop('host.distance'),
                snap['gas'].prop('host.distance')))

all_mass = np.concatenate((snap['star']['mass'][ss_key],
                snap['dark']['mass'],
                snap['gas']['mass']))

Rmag = np.linalg.norm(all_position, axis=1)
Rkeys = np.where(Rmag < Rmax)[0]

all_position = all_position[Rkeys]
all_mass = all_mass[Rkeys]
all_softening = all_softening[Rkeys]

grav_code = pykdgrav(all_position, all_mass, all_softening, theta, G)

