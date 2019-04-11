import gizmo_analysis as gizmo
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 512
npix = hp.nside2npix(nside)

observer_position = np.array([8.2, 0, 0])

def read_snap(gal):
    # takes in galaxy (string = m12i, m12f, m12m, etc.)
    # reads in and sets fiducial coordinates
    # returns snap
    gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'
    # sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/'+gal+'_res7100'
    sim_directory = '../data/fire2/metaldiff/'+gal+'_res7100'
    # snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600,
    snap = gizmo.io.Read.read_snapshots(['star'], 'index', 600,
                                        properties=['position', 'id',
                                                    'mass'],
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

    return snap

def get_flux(snap, obs_pos):
    star_pos = snap['star'].prop('host.distance.principal')
    star_mass = snap['star']['mass']

    rel_pos = np.subtract(star_pos, obs_pos)
    rmag = np.linalg.norm(rel_pos, axis=1)
    theta = np.arccos(rel_pos[:,2]/rmag)
    phi = np.arctan2(rel_pos[:,1], rel_pos[:,0])

    phi = np.mod(phi + np.pi, 2.*np.pi)

    flux = star_mass/np.square(rmag)

    return theta, phi, flux

for gal in ['m12i', 'm12f', 'm12m']:
    snap = read_snap(gal)

    wisemap = np.zeros(npix)

    theta, phi, flux = get_flux(snap, observer_position)

    ipix = hp.ang2pix(nside, theta, phi)
    for idx,k in enumerate(ipix):
        wisemap[k] += flux[idx]

    wisemap = np.log10(wisemap)

    hp.mollview(wisemap, min=1, max=4)
    plt.title(gal)
    plt.savefig('WISE_'+gal+'.pdf')
