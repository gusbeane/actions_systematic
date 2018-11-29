import numpy as np
from amuse.units import units, constants, nbody_system
from amuse.ext.bridge import bridge
from amuse.community.ph4.interface import ph4
from amuse.community.fi.interface import Fi
from amuse.community.bhtree.interface import BHTree
from amuse.community.gadget2.interface import Gadget2
import matplotlib.pyplot as plt
from amuse.ic.kingmodel import new_king_model
from amuse.community.galaxia.interface import BarAndSpirals3D
from tqdm import tqdm

import gala.potential as gp
import gala.dynamics as gd
import astropy.units as u
from astropy.table import Table

import pickle

class GalaxyGravityCode(object):
    def __init__(self):
        self.mw = gp.MilkyWayPotential()

    def get_gravity_at_point(self, eps, x, y, z):
        pos = [x.value_in(units.kpc), y.value_in(units.kpc), z.value_in(units.kpc)]
        acc = self.mw.acceleration(pos).to_value(u.km/u.s/u.Myr)
        return acc | units.km/units.s/units.Myr

    def circular_velocity(self, x, y, z):
        pos = [x.value_in(units.kpc), y.value_in(units.kpc), z.value_in(units.kpc)]
        print pos
        vc = self.mw.circular_velocity(pos).to_value(u.km/u.s)
        return vc | units.kms
    
    def get_potential_at_point(self, eps, x, y, z):
        pos = [x.value_in(units.kpc), y.value_in(units.kpc), z.value_in(units.kpc)]
        pot = self.mw.energy(pos).value_to(u.kpc**2/u.Myr**2)
        return pot | units.kpc**2 / units.Myr**2

def make_king_model_cluster(nbodycode, N, W0, Mcluster,
                            Rcluster, parameters = []):
    
    converter = nbody_system.nbody_to_si(Mcluster, Rcluster)
    bodies = new_king_model(N, W0, convert_nbody=converter)

    
    code = nbodycode(converter)
    #code = nbodycode(converter)
    for name, value in parameters:
        setattr(code.parameters, name, value)
    code.particles.add_particles(bodies)
    return code

def store_frame(t, x, y, z, vx, vy, vz):
    xo = x.value_in(units.parsec).tolist()
    yo = y.value_in(units.parsec).tolist()
    zo = z.value_in(units.parsec).tolist()
    vxo = vx.value_in(units.kms).tolist()
    vyo = vy.value_in(units.kms).tolist()
    vzo = vz.value_in(units.kms).tolist()
    meta = {"time": t.value_in(units.Myr) * u.Myr}
    t = Table([xo, yo, zo, vxo, vyo, vzo], meta=meta, names=['posx', 'posy', 'posz', 'velx', 'vely', 'velz'])
    t['posx'].unit = u.pc
    t['posy'].unit = u.pc
    t['posz'].unit = u.pc
    t['velx'].unit = u.km/u.s
    t['vely'].unit = u.km/u.s
    t['velz'].unit = u.km/u.s
    return t

def plot_cluster(xlist, ylist):
    
    x = xlist[-1]
    y = ylist[-1]

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.scatter(x,y, c='k', s=1, lw=0)
    
    save_file = 'Arches_Fig_7.1.png'
    plt.savefig(save_file)
    print '\nSaved figure in file', save_file, '\n'
    plt.show()

def evolve_cluster_in_galaxy(N, W0, Rinit, tend, timestep, M, R):
    galaxy_code = GalaxyGravityCode()

    cluster_code = make_king_model_cluster(ph4, N, W0, M, R, parameters=[("epsilon_squared", (0.01 | units.parsec)**2)])
    
    stars = cluster_code.particles.copy()
    stars.x += Rinit
    stars.vy = 0.8*galaxy_code.circular_velocity(Rinit, 0 | units.kpc, 0 | units.kpc)
    channel = stars.new_channel_to(cluster_code.particles)
    channel.copy_attributes(["x","y","z","vx","vy","vz"])

    system = bridge(verbose=False)
    system.add_system(cluster_code, (galaxy_code,), False)
    
    x = [system.particles.x.value_in(units.kpc)]
    y = [system.particles.y.value_in(units.kpc)]
    
    times = np.arange(0|units.Myr, tend, timestep)
    output = []
    for i,t in enumerate(tqdm(times)):
        system.evolve_model(t,timestep=timestep)
        x.append(system.particles.x.value_in(units.kpc))
        y.append(system.particles.y.value_in(units.kpc))
        if i%5 == 0:
            output.append(store_frame(t, system.particles.x, system.particles.y, system.particles.z, system.particles.vx, system.particles.vy, system.particles.vz))

    cluster_code.stop()
    
    f=open("aximw.p", "wb")
    pickle.dump(output, f)
    f.close()

    return x, y

def make_movie(xlist, ylist):
    import matplotlib.animation as manimation
    
    fig = plt.figure(figsize=(3,3))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='cluster_galaxy_center', artist='abeane')
    writer = FFMpegWriter(fps=30, metadata=metadata, bitrate=10)
    
    def init_fig():
        fig.clear()
        plt.rcParams.update({'font.size': 10})
        plot = fig.add_subplot(1,1,1)
        ax = plt.gca()
        ax.minorticks_on()
        ax.locator_params(nbins=3)

        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plot.set_xlim(-9, 9)
        plot.set_ylim(-9, 9)
        return plot

    print 'making movie...'
    with writer.saving(fig, "cluster_galaxy_center.mp4", len(xlist)):
        for i in tqdm(range(len(xlist))):
            plot = init_fig()       
            plot.scatter(xlist[i],ylist[i],c='k', s=1)
            writer.grab_frame()
    return None

if __name__ == '__main__':
    N = 1024
    W0 = 3
    Rinit= 8. | units.kpc
    timestep=0.1 | units.Myr
    endtime = 250 | units.Myr
    Mcluster = 5.e4 | units.MSun
    Rcluster = 0.8 | units.parsec
    x, y = evolve_cluster_in_galaxy(N, W0, Rinit, endtime, timestep, Mcluster, Rcluster)
    
    #plot_cluster(x, y)
    make_movie(x, y)
