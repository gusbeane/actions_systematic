import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm

sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/m12i_res7100'
snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600,
                                   properties=['position', 'id',
                                               'mass', 'velocity',
                                               'form.scalefactor'],
                                     assign_principal_axes=True,
                                     simulation_directory=sim_directory)

star_pos = snap['star'].prop('host.distance.principal')

R = 8.2
theta = np.linspace(0, 2.*np.pi, 20)
posx = R*np.cos(theta)
posy = R*np.sin(theta)
posz = np.zeros(len(posx))
pos = np.transpose([posx, posy, posz])

rcut = 0.20
zcut = 1.0
def get_keys(p):
    pos_diff = np.subtract(star_pos, p)
    rmag = np.linalg.norm(pos_diff[:,:2], axis=1)
    rbool = rmag < rcut
    zbool = np.abs(pos_diff[:,2]) < zcut
    keys = np.where(np.logical_and(rbool, zbool))[0]
    return keys

midplane_est = []
for p in pos:
    new_p = p.copy()
    for _ in range(10):
        keys = get_keys(new_p)
        print(len(keys))
        new_p[2] = np.median(star_pos[:,2][keys])
    midplane_est.append(new_p[2])

    to_bootstrap = star_pos[:,2][keys]

