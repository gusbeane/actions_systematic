import sys
sys.path.append('../')
import numpy as np

from schmactions import schmactions
from schmactions import compute_offset_list

import astropy.units as u
import pickle

from joblib import Parallel, delayed
import multiprocessing

nproc = 40

init_pos = [8, 0, 0] * u.kpc
init_vel_list = [ [0, -190, v] * u.km/u.s for v in np.linspace(0, 200, 100) ]

zoffset = [0, 0, 100] * u.pc
voffset = [0, 0, 0] * u.km/u.s

def _helper_(init_vel):
    s = schmactions(init_pos, init_vel)
    r = s.compute_actions_offset(zoffset, voffset)
    act = s.extract_actions(r)

    true_actions = s.res['actions'].to_value(u.kpc * u.km/u.s)

    return [true_actions, act]

out = Parallel(n_jobs=nproc) (delayed(_helper_)(init_vel) for init_vel in tqdm(init_vel_list))

pickle.dump(out, open('dJz_fun_of_Jz.p', 'wb'))

