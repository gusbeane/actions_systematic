import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

nbootstrap = 1000
np.random.seed(162)

gal_info = 'oa_frame.txt'

sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/m12i_res7100'
snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600,
                                   properties=['position', 'id',
                                               'mass', 'velocity',
                                               'form.scalefactor'],
                                     assign_center=False,
                                     simulation_directory=sim_directory)

ref = np.genfromtxt(gal_info, comments='#')
snap.center_position = ref[0]
snap.center_velocity = ref[1]
snap.principal_axes_vectors = ref[2:]
for k in snap.keys():
    for attr in ['center_position', 'center_velocity', 
                 'principal_axes_vectors']:
        setattr(snap[k], attr, getattr(snap, attr))

star_pos = snap['star'].prop('host.distance.principal')

R = 8.2
theta = np.linspace(0, 2.*np.pi, 100)
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
err_low = []
err_high = []
for p in pos:
    new_p = p.copy()
    for _ in range(10):
        keys = get_keys(new_p)
        new_p[2] = np.median(star_pos[:,2][keys])
    midplane_est.append(new_p[2])

    central_val = new_p[2]

    to_bootstrap = star_pos[:,2][keys]
    keys_to_choose = list(range(len(to_bootstrap)))
    rand_choice = np.random.choice(keys_to_choose, len(to_bootstrap)*nbootstrap)

    heights_rand = to_bootstrap[rand_choice]
    heights_rand = np.reshape(heights_rand, (nbootstrap, len(to_bootstrap)))
    med_rand = np.median(heights_rand, axis=1)
    dist = np.subtract(med_rand, central_val)
    up = np.percentile(dist, 95)
    low = np.percentile(dist, 5)
    err_low.append(central_val - up)
    err_high.append(central_val - low)
    print(central_val - up, central_val - low)

midplane_est = np.array(midplane_est)
err_low = np.array(err_low)
err_high = np.array(err_high)

fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(theta/np.pi, midplane_est*1000, c=tb_c[0])
ax.plot(theta/np.pi, err_low*1000, c=tb_c[0], ls='dashed')
ax.plot(theta/np.pi, err_high*1000, c=tb_c[0], ls='dashed')
ax.fill_between(theta/np.pi, err_high*1000, err_low*1000, color=tb_c[0], alpha=0.5)

ax.set_xlabel(r'$\theta/\pi$')
ax.set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')

ax.set_xlim(0, 2)

fig.tight_layout()

plt.savefig('midplane.pdf')
