import gizmo_analysis as gizmo
import numpy as np 
import agama

# first read in snapshot

np.random.seed(162)

snap_idx=600
sim_dir = '/mnt/ceph/users/firesims/fire2/metaldiff/m12i_res7100/'
gal_info = 'm12i_info.txt'

Jr_min_list = [10.0, 40.0]
Jr_max_list = [20.0, 60.0]
Jz_min_list = [10.0, 40.0]
Jz_max_list = [20.0, 60.0]

Rmin = 7.7
Rmax = 8.7
zmax = 0.5

age_min = 0.25
age_max = 0.75

act_cuts = zip(Jr_min_list, Jr_max_list, Jz_min_list, Jz_max_list)

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

star_fid_pos = snap['star'].prop('host.distance.principal')
gas_fid_pos = snap['gas'].prop('host.distance.principal')
dark_fid_pos = snap['dark'].prop('host.distance.principal')

star_fid_vel = snap['star'].prop('host.velocity.principal')

star_mass = snap['star']['mass']
gas_mass = snap['gas']['mass']
dark_mass = snap['dark']['mass']

star_age = snap['star'].prop('age')

position = np.concatenate((star_fid_pos, gas_fid_pos))
mass = np.concatenate((star_mass, gas_mass))

pdark = agama.Potential(type="Multipole",
                        particles=(dark_fid_pos, dark_mass),
                        symmetry='a', gridsizeR=20, lmax=2)
pbar = agama.Potential(type="CylSpline",
                       particles=(position, mass),
                       gridsizer=20, gridsizez=20,
                       mmax=0, Rmin=0.2, symmetry='a',
                       Rmax=50, Zmin=0.02, Zmax=10)

potential = agama.Potential(pdark, pbar)
af = agama.ActionFinder(potential, interp=False)

star_fid_phase = np.hstack((star_fid_pos, star_fid_vel))
actions = af(star_fid_phase)
for Jr_min, Jr_max, Jz_min, Jz_max in act_cuts:
    Jrbool = np.logical_and(actions[:,0] > Jr_min, actions[:,0] < Jr_max)
    Jzbool = np.logical_and(actions[:,1] > Jz_min, actions[:,1] < Jz_max)

    R = np.linalg.norm(star_fid_pos[:,:2], axis=1)
    Rbool = np.logical_and(R > Rmin, R < Rmax)
    zbool = np.abs(star_fid_pos[:,2]) < zmax

    agebool = np.logical_and(star_age > age_min, star_age < age_max)

    posbool = np.logical_and(Rbool, np.logical_and(zbool, agebool))
    actbool = np.logical_and(Jrbool, Jzbool)
    keys = np.where(np.logical_and(posbool, actbool))[0]
    chosen_key = np.random.choice(keys)
    chosen_id = snap['star']['id'][chosen_key]

    print(Jr_min, Jr_max, Jz_min, Jz_max, chosen_id)
