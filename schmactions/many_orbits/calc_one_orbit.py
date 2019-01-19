import sys
sys.path.append('../')

from schmactions import schmactions
from schmactions import compute_offset_list

import astropy.units as u
import pickle

nproc = 40
init_pos = [8, 0, 0] * u.kpc
init_vel = [0, -190, 50] * u.km/u.s

zoffset_list = [[0, 0, z] * u.pc for z in np.arange(0, 500, 10)]
xoffset_list = [[x, 0, 0] * u.pc for x in np.arange(0, 500, 10)]

voffset = [0, 0, 0] * u.km/u.s

s = schmactions(init_pos, init_vel)

pickle.dump(s.res, open('true_res.p', 'wb')) 

zout = s.compute_offset_list(zoffset_list, nproc=nproc)
xout = s.compute_offset_list(xoffset_list, nproc=nproc)

pickle.dump(zout, open('zout.p', 'wb'), protocol=4)
pickle.dump(xout, open('xout.p', 'wb'), protocol=4)
