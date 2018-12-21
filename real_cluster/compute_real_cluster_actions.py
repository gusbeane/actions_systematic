import numpy as np 

import gala.dynamics as gd
import gala.potential as gp
import gala.integrate as gi

from pyia import GaiaData
from astropy.table import Table 
import astropy.units as u
import astropy.coordinates as coord

import warnings

dt = 1 * u.Myr
mw = gp.MilkyWayPotential()

t = Table.read('J_A+A_616_A10_tablea3.dat.fits', format='fits')

# rename stuff
t['ra'] = t['RAdeg']
t['dec'] = t['DEdeg']
t['parallax'] = t['plx']
t['pmra'] = t['pmRA']
t['pmdec'] = t['pmDE']
t['radial_velocity'] = t['RV']

g = GaiaData(t)

dist = 1/(g.parallax.to_value(u.mas)) * 1000 # pc
name = t['Cluster'].tolist()
m = np.c_[name, dist]

rsun = 8.2 * u.kpc
zsun = 0.025 * u.kpc
vsun = [11.1, 232.24, 7.25] * u.km/u.s
gc_frame = coord.Galactocentric(galcen_distance=rsun, galcen_v_sun=coord.CartesianDifferential(*vsun), z_sun=zsun)

sc = g.skycoord
scdyn = gd.PhaseSpacePosition(sc.transform_to(gc_frame).cartesian)

def actions(star):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        orbit = mw.integrate_orbit(star, dt=dt, t1=0*u.Gyr, t2=5*u.Gyr, Integrator=gi.DOPRI853Integrator)
        res = gd.actionangle.find_actions(orbit, N_max=8)
        ans = res['actions'].to(u.kpc * u.km / u.s).value
        return np.append(ans,orbit.zmax(approximate=True).to(u.pc).value)

result = [actions(s) for s in scdyn]
result2 = np.append((m, result), axis=1)
r = Table(result2, names=('cluster', 'distance', 'Jr', 'Lz', 'Jz', 'zmax'))
r['distance'].unit = u.pc
r['Jr'].unit = u.kpc * u.km/u.s
r['Lz'].unit = u.kpc * u.km/u.s
r['Jz'].unit = u.kpc * u.km/u.s
r['zmax'].unit = u.pc

r.write('real_cluster_actions.fits', format='fits')
r.write('real_cluster_actions.tex', format='ascii.aastex')