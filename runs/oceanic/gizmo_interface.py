import matplotlib; matplotlib.use('agg')

import numpy as np
import gizmo_analysis as gizmo
from amuse.units import units
from oceanic.options import options_reader
from oceanic.analysis import agama_wrapper
import pickle

from pykdgrav import ConstructKDTree, GetAccelParallel
from astropy.constants import G as G_astropy
import astropy.units as u

import sys
import os


class gizmo_interface(object):
    def __init__(self, options_reader, grid_snapshot=None):

        self.G = G_astropy.to_value(u.kpc**2 * u.km / (u.s * u.Myr * u.Msun))
        self.theta = 0.5

        self.convert_kms_Myr_to_kpc = 20000.0*np.pi / (61478577.0)
        self.kpc_in_km = 3.08567758E16
        # opt = options_reader(options_file)
        options_reader.set_options(self)
        self.options_reader = options_reader

        if not os.path.isdir(self.cache_directory):
            os.makedirs(self.cache_directory)

        self._read_snapshots_()

        # find starting star
        if self.axisymmetric:
            if self.axi_Rinit is None or self.axi_vcircfrac is None or self.axi_zinit is None:
                self._init_starting_star_()
            self._gen_axisymmetric_()
        else:
            self._init_starting_star_()

        self._init_kdtree_()

        self.evolve_model(0 | units.Myr)

    def _read_snapshots_(self):
        # read in first snapshot, get rotation matrix

        # just gonna take a peak into the sim and see if we have it in cache
        head = gizmo.io.Read.read_header(snapshot_value=self.snap_index,
                                         simulation_directory=
                                         self.simulation_directory)
        if self.sim_name is None:
            self.sim_name = head['simulation.name'].replace(" ", "_")
        cache_name = 'first_snapshot_' + self.sim_name+'_index' + \
            str(self.snap_index)+'.p'
        cache_file = self.cache_directory + '/' + cache_name

        try:
            self.first_snapshot = pickle.load(open(cache_file, 'rb'))
            print('found and loaded cached file for first_snapshot:')
            print(cache_name)

        except:
            print('couldnt find cached file for first_snapshot:')
            print(cache_name)
            print('constructing...')
            self.first_snapshot =\
                gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                             'index', self.snap_index,
                                             simulation_directory=
                                             self.simulation_directory,
                                             assign_center=False)
            pickle.dump(self.first_snapshot,
                        open(cache_file, 'wb'), protocol=4)

        if self.gal_info is not None:
            gal = np.genfromtxt(self.gal_info, comments='#', delimiter=',')
            cen = gal[0]
            vel = gal[1]
            pa = gal[2:]
            self.first_snapshot.center_position = cen
            self.first_snapshot.center_velocity = vel
            self.first_snapshot.principal_axes_vectors = pa
            for k in self.first_snapshot.keys():
                self.first_snapshot[k].center_position = cen
                self.first_snapshot[k].center_velocity = vel
                self.first_snapshot[k].principal_axes_vectors = pa
        else:
            gizmo.io.Read.assign_center(self.first_snapshot)
            gizmo.io.Read.assign_principal_axes(self.first_snapshot, distance_max=20, age_percent=100)
        
        self.center_position = self.first_snapshot.center_position
        self.center_velocity = self.first_snapshot.center_velocity
        self.principal_axes_vectors =\
            self.first_snapshot.principal_axes_vectors

        # store some other relevant information
        self.first_snapshot_time_in_Myr =\
            self.first_snapshot.snapshot['time'] * 1000.0

        self._clean_Rmag_(self.first_snapshot)

    def _gen_axisymmetric_(self):
        import agama
        potential_cache_file = self.cache_directory + '/potential_id'+str(self.snap_index)
        potential_cache_file += '_' + self.sim_name + '_pot'
        try:
            self.potential = agama.Potential(file=potential_cache_file)
        except:
            star_position = self.first_snapshot['star'].prop('host.distance.principal')
            gas_position = self.first_snapshot['gas'].prop('host.distance.principal')
            dark_position = self.first_snapshot['dark'].prop('host.distance.principal')

            star_mass = self.first_snapshot['star']['mass']
            gas_mass = self.first_snapshot['gas']['mass']
            dark_mass = self.first_snapshot['dark']['mass']

            position = np.concatenate((star_position, gas_position))
            mass = np.concatenate((star_mass, gas_mass))

            #TODO make these user-controllable
            self.pdark = agama.Potential(type="Multipole",
                                        particles=(dark_position, dark_mass),
                                        symmetry='a', gridsizeR=20, lmax=2)
            self.pbar = agama.Potential(type="CylSpline",
                                        particles=(position, mass),
                                        symmetry='a', gridsizer=20, gridsizez=20,
                                        mmax=0, Rmin=0.2,
                                        Rmax=50, Zmin=0.02, Zmax=10)
            self.potential = agama.Potential(self.pdark, self.pbar)
            self.potential.export(potential_cache_file)
        return None


    def _clean_Rmag_(self, snap):
        # cleans out all particles greater than Rmag from galactic center
        for key in snap.keys():
            rmag = snap[key].prop('host.distance.total')
            rmag_keys = np.where(rmag < self.Rmax)[0]
            for dict_key in snap[key].keys():
                snap[key][dict_key] = snap[key][dict_key][rmag_keys]
        return snap

    def _init_kdtree_(self, snap=None):
        if snap is None:
            snap = self.first_snapshot
        
        # first exclude starting star
        ss_key = np.where(snap['star']['id'] != self.chosen_id)[0]

        # gather all necessary parameters
        all_position = np.concatenate((snap['star'].prop('host.distance.principal')[ss_key],
                snap['dark'].prop('host.distance.principal'),
                snap['gas'].prop('host.distance.principal')))

        # all_velocity = np.concatenate((snap['star'].prop('host.velocity.principal')[ss_key],
        #         snap['dark'].prop('host.velocity.principal'),
        #         snap['gas'].prop('host.velocity.principal')))

        all_mass = np.concatenate((snap['star']['mass'][ss_key],
                snap['dark']['mass'],
                snap['gas']['mass']))

        # set star softening
        # the M^(1/3) param comes from a tidal force argument, see paper
        if self.star_char_mass is not None:
            star_mass = snap['star']['mass']
            star_softening = np.power(star_mass/self.star_char_mass, 1.0/3.0)
            star_softening /= 1000.0
        else:
            star_softening = np.full(len(snap['star']['position']),
                            float(self.star_softening_in_pc)/1000.0)[ss_key]

        # same but for dark matter
        if self.dark_char_mass is not None:
            dark_mass = snap['dark']['mass']
            dark_softening = np.power(dark_mass/self.dark_char_mass, 1.0/3.0)
            dark_softening /= 1000.0
        else:
            dark_softening = np.full(len(snap['dark']['position']),
                float(self.dark_softening_in_pc)/1000.0)

        # we don't need to do anything fancy for the gas
        gas_softening = 2.8 * snap['gas']['smooth.length'] / 1000.0

        all_softening = np.concatenate((star_softening, dark_softening,
                                        gas_softening))

        r = all_position
        m = all_mass
        soft = all_softening

        print('constructing tree for gravity calculation')
        self.tree = ConstructKDTree( np.float64(r), np.float64(m), np.float64(soft))
        print('done constructing tree')
        return None 

    def evolve_model(self, time, timestep=None):
        this_t_in_Myr = time.value_in(units.Myr)
        self._time_ = this_t_in_Myr

    def get_gravity_at_point(self, eps, xlist, ylist, zlist):
        # convert to kpc
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)
        
        if self.axisymmetric:
            pos = np.transpose([xlist, ylist, zlist])
            acc = self.potential.force(pos)
            if len(np.shape(pos))==1:
                ax = acc[0] | (units.kms)**2/units.kpc
                ay = acc[1] | (units.kms)**2/units.kpc
                az = acc[2] | (units.kms)**2/units.kpc
            else:
                ax = acc[:,0] | (units.kms)**2/units.kpc
                ay = acc[:,1] | (units.kms)**2/units.kpc
                az = acc[:,2] | (units.kms)**2/units.kpc
            return ax, ay, az

        position = np.transpose([xlist, ylist, zlist])
        single = False
        if len(np.shape(position)) == 1:
            position = np.array([position])
            single = True

        # rotate the cluster to where the galaxy has moved
        position = self._rotate_position_(position, self._time_)

        accel = GetAccelParallel(position, self.tree, self.G, self.theta)
        
        # need to rotate back
        accel = self._rotate_gravity_(accel, self._time_)
        
        ax = accel[:,0] | units.kms/units.Myr
        ay = accel[:,1] | units.kms/units.Myr
        az = accel[:,2] | units.kms/units.Myr
        return ax, ay, az

    def _rotate_gravity_(self, acc, t):
        # this is the same as rotate_position, but with a different sign for angle
        if self.period is None:
            return acc
        angle = (t/self.period) * 2.*np.pi
        rotmat = self._rotmat_(angle)
        
        rot_acc = np.transpose(np.tensordot(rotmat, np.transpose(acc), axes=1))
        return rot_acc

    def _rotate_position_(self, pos, t):
        if self.period is None:
            return pos
        # we want to rotate the cluster back
        # since we're keeping the galaxy fixed (so we don't have to recompute the tree)
        angle = -(t/self.period) * 2.*np.pi
        rotmat = self._rotmat_(angle)
        


        # this applies the matrix transformation to all positions
        # don't ask...
        rot_pos = np.transpose(np.tensordot(rotmat, np.transpose(pos), axes=1))
        return rot_pos

    def _rotmat_(self, angle):
        ct = np.cos(angle)
        st = np.sin(angle)
        mat = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
        return mat


    # def get_tidal_tensor_at_point(self, eps, xlist, ylist, zlist):
    #     # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
    #     xlist = xlist.value_in(units.kpc)
    #     ylist = ylist.value_in(units.kpc)
    #     zlist = zlist.value_in(units.kpc)

    #     if hasattr(xlist, '__iter__'):
    #         Tlist = []
    #         for x, y, z in zip(xlist, ylist, zlist):
    #             rbfi_x, rbfi_y, rbfi_z = self._get_acc_rbfi_(x, y, z)
    #             Txx = float(rbfi_x([[x, y, z]], diff=(1, 0, 0)))
    #             Tyy = float(rbfi_y([[x, y, z]], diff=(0, 1, 0)))
    #             Tzz = float(rbfi_z([[x, y, z]], diff=(0, 0, 1)))
    #             Txy = float(rbfi_y([[x, y, z]], diff=(1, 0, 0)))
    #             Tyx = float(rbfi_x([[x, y, z]], diff=(0, 1, 0)))
    #             Txz = float(rbfi_z([[x, y, z]], diff=(1, 0, 0)))
    #             Tzx = float(rbfi_x([[x, y, z]], diff=(0, 0, 1)))
    #             Tyz = float(rbfi_z([[x, y, z]], diff=(0, 1, 0)))
    #             Tzy = float(rbfi_y([[x, y, z]], diff=(0, 0, 1)))
    #             T = [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
    #             Tlist.append(T)
    #         # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
    #         return Tlist | units.kms/units.Myr/units.kpc

    #     else:
    #         rbfi_x, rbfi_y, rbfi_z = self._get_acc_rbfi_(xlist, ylist, zlist)
    #         # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
    #         Txx = float(rbfi_x([[xlist, ylist, zlist]], diff=(1, 0, 0)))
    #         Tyy = float(rbfi_y([[xlist, ylist, zlist]], diff=(0, 1, 0)))
    #         Tzz = float(rbfi_z([[xlist, ylist, zlist]], diff=(0, 0, 1)))
    #         Txy = float(rbfi_y([[xlist, ylist, zlist]], diff=(1, 0, 0)))
    #         Tyx = float(rbfi_x([[xlist, ylist, zlist]], diff=(0, 1, 0)))
    #         Txz = float(rbfi_z([[xlist, ylist, zlist]], diff=(1, 0, 0)))
    #         Tzx = float(rbfi_x([[xlist, ylist, zlist]], diff=(0, 0, 1)))
    #         Tyz = float(rbfi_z([[xlist, ylist, zlist]], diff=(0, 1, 0)))
    #         Tzy = float(rbfi_y([[xlist, ylist, zlist]], diff=(0, 0, 1)))
    #         T = [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
    #         return T | units.kms/units.Myr/units.kpc

    # TODO clean up starting star
    def _init_starting_star_(self):
        # TEST
        #self.chosen_id = 10
        #return None # just for testing
        
        # END TEST

        self.chosen_position_z0, self.chosen_velocity_z0, self.chosen_index_z0, self.chosen_id = \
                            self.starting_star(self.ss_Rmin, self.ss_Rmax,
                                               self.ss_zmin, self.ss_zmax,
                                               self.ss_agemin_in_Gyr,
                                               self.ss_agemax_in_Gyr,
                                               self.ss_seed)

    def starting_star(self, Rmin, Rmax, zmin, zmax, agemin_in_Gyr,
                      agemax_in_Gyr, seed=1776):
            if self.ss_id is not None:
                pos = self.first_snapshot['star'].prop('host.distance.principal')
                chosen_one = np.where(self.first_snapshot['star']['id'] == \
                                      self.ss_id)[0]
                return pos[chosen_one], chosen_one, self.ss_id

            np.random.seed(seed)

            starages = self.first_snapshot['star'].prop('age')
            pos = self.first_snapshot['star'].prop('host.distance.principal')
            # vel = self.first_snapshot['star'].prop('host.velocity.principal')

            Rstar = np.sqrt(pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1])
            zstar = pos[:, 2]

            agebool = np.logical_and(starages > agemin_in_Gyr,
                                     starages < agemax_in_Gyr)
            Rbool = np.logical_and(Rstar > Rmin, Rstar < Rmax)
            zbool = np.logical_and(zstar > zmin, zstar < zmax)

            totbool = np.logical_and(np.logical_and(agebool, Rbool), zbool)
            keys = np.where(totbool)[0]

            if self.ss_action_cuts:
                np.random.seed(seed)
                starages = self.first_snapshot['star'].prop('age')[keys]
                pos = self.first_snapshot['star'].prop('host.distance.principal')[keys]
                self.first_ag = agama_wrapper(self.options_reader)
                self.first_ag.update_index(self.snap_index, snap=self.first_snapshot)
                for ss_id in np.random.permutation(self.first_snapshot['star']['id'][keys]):
                    self.first_ag.update_ss(ss_id)
                    self.chosen_actions = self.first_ag.ss_action()
                    print(self.chosen_actions)
                    Jr = self.chosen_actions[0]
                    Jz = self.chosen_actions[1]
                    # Lz = self.chosen_actions[2]
                    Jrbool = Jr > self.Jr_min and Jr < self.Jr_max
                    Jzbool = Jz > self.Jz_min and Jz < self.Jz_max
                    if Jrbool and Jzbool:
                        pos = self.first_snapshot['star'].\
                                prop('host.distance.principal')
                        vel = self.first_snapshot['star'].\
                                prop('host.velocity.principal')
                        chosen_one = np.where(self.first_snapshot['star']['id']
                                              == ss_id)[0]
                        print('pos: ', pos[chosen_one])
                        print('vel: ', vel[chosen_one])
                        print('chosen_one: ', chosen_one)
                        print('ss_id: ', ss_id)
                        return pos[chosen_one][0], vel[chosen_one][0], int(chosen_one), ss_id

            chosen_one = np.random.choice(keys)
            chosen_id = self.first_snapshot['star']['id'][chosen_one]
            # return pos[chosen_one], vel[chosen_one], chosen_one, chosen_id
            return pos[chosen_one], chosen_one, chosen_id

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['acc_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    options_file = sys.argv[1]
    opt = options_reader(options_file)

    if len(sys.argv) == 3:
        snap_index = int(sys.argv[2])
        g = gizmo_interface(opt, snap_index)
    else:
        g = gizmo_interface(opt)
