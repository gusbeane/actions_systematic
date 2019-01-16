import numpy as np

import astropy.units as u

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp

from tqdm import tqdm
import warnings

class schmactions(object):
    def __init__(self, init_pos, init_vel, mw=None,
                 t1=0*u.Gyr, t2=5*u.Gyr, dt=1*u.Myr, N_max=8):
        
        if mw is None:
            self.mw = gp.MilkyWayPotential()
        else:
            self.mw = mw

        self.t1 = t1
        self.t2 = t2
        self.dt = dt
        self.N_max = 8
        
        self.q = gd.PhaseSpacePosition(pos=init_pos, vel=init_vel)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            self.orbit = self.mw.integrate_orbit(self.q, dt=self.dt,
                                                 t1=self.t1, t2=self.t2)
            self.res = gd.find_actions(self.orbit, N_max=self.N_max)

    def compute_actions_offset(self, pos_offset, vel_offset, 
                               cadence=10, end=1000):
        pos = np.transpose(self.orbit.pos.xyz)
        pos = np.add(pos, pos_offset)
        vel = np.transpose(self.orbit.vel.d_xyz)
        vel = np.add(vel, vel_offset)

        pos = np.transpose(pos)
        vel = np.transpose(vel)

        all_q = gd.PhaseSpacePosition(pos=pos, vel=vel)
        out = []
        time = self.orbit.t
        with warnings.catch_warnings(record=True):
            for q,t in tqdm(zip(all_q[0:end:cadence],time[0:end:cadence])):
                orbit = self.mw.integrate_orbit(q, dt=self.dt,
                                                t1=self.t1, t2=self.t2)
                res = gd.find_actions(orbit, N_max=self.N_max)
                res['time'] = t
                out.append(res)
        return out

