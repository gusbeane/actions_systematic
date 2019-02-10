import numpy as np
import gizmo_analysis as gizmo

def read_snap(gal):
    # takes in galaxy (string = m12i, m12f, m12m, etc.)
    # reads in and sets fiducial coordinates
    # returns snap
    gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'
    sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/'+gal+'_res7100'
    snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600,
                                        properties=['position', 'id',
                                                    'mass', 'velocity',
                                                    'form.scalefactor',
                                                    'smooth.length'],
                                        assign_center=False,
                                        simulation_directory=sim_directory)
    ref = np.genfromtxt(gal_info, comments='#', delimiter=',')
    snap.center_position = ref[0]
    snap.center_velocity = ref[1]
    snap.principal_axes_vectors = ref[2:]
    for k in snap.keys():
        for attr in ['center_position', 'center_velocity', 
                     'principal_axes_vectors']:
            setattr(snap[k], attr, getattr(snap, attr))

    return snap

def get_lsr(pos, star_pos, star_vel):
    # first get all star particles within rcut_in_pc of pos
    rcut = rcut_in_pc/1000
    diff = np.subtract(star_pos, pos)
    diff_mag = np.linalg.norm(diff, axis=1)

    keys = np.where(np.less(diff_mag, rcut))[0]
    star_select = star_vel[keys]
    lsr = np.median(star_select, axis=0)
    return lsr

if __name__ == '__main__':
    rcut_in_pc = 200 # rcut for lsr in pc
    Rsolar = 8.2
    thetalist = np.linspace(0, 2.*np.pi, 50)

    for gal in ['m12i', 'm12f', 'm12m']:
        snap = read_snap(gal)
        star_pos = snap['star'].prop('host.distance.principal')
        star_vel = snap['star'].prop('host.velocity.principal.cylindrical')
        
        poslist = np.array( [[Rsolar*np.cos(t), Rsolar*np.sin(t), 0] for t in thetalist] )
        out = np.concatenate((thetalist, lsrlist), axis=1)
        lsrlist = np.array( [get_lsr(pos, star_pos, star_vel) for pos in tqdm(poslist)] )
        np.save('output/lsr_'+gal+'.npy', out)
