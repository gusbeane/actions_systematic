import sys
sys.path.append('../')
import numpy as np

from schmactions import schmactions
from schmactions import compute_offset_list

import astropy.units as u
import pickle
from astropy.table import Table

nproc = 40

t = Table.read('real_cluster_gc.fits', format='fits')

posx = t['pos.x'].astype('float').tolist()
posy = t['pos.y'].astype('float').tolist()
posz = t['pos.z'].astype('float').tolist()
velx = t['vel.x'].astype('float').tolist()
vely = t['vel.y'].astype('float').tolist() 
velz = t['vel.z'].astype('float').tolist()

init_pos_list = np.transpose([posx, posy, posz]) * u.pc
init_vel_list = np.transpose([velx, vely, velz]) * u.km/u.s

name_list = t['cluster']

zoffset_list = [[0, 0, z] * u.pc for z in np.arange(0, 500, 10)]
xoffset_list = [[x, 0, 0] * u.pc for x in np.arange(0, 500, 10)]

voffset = [0, 0, 0] * u.km/u.s

for init_pos, init_vel, name in zip(init_pos_list, init_vel_list, name_list):
    s = schmactions(init_pos, init_vel)

    pickle.dump(s.res, open('true_res_'+name+'.p', 'wb'))

    zout = compute_offset_list(s, zoffset_list, nproc=nproc)
    xout = compute_offset_list(s, xoffset_list, nproc=nproc)

    pickle.dump(zout, open('zout_'+name+'.p', 'wb'), protocol=4)
    pickle.dump(xout, open('xout_'+name+'.p', 'wb'), protocol=4)
