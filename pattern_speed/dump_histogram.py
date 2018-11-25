import numpy as np
import gizmo_analysis as gizmo

start = 591
end = 600
snap_ids = list(range(start, end+1))

rmin = 7.2
rmax = 9.2
zmax = 1.0

for gal in ['m12i', 'm12f']:
    sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/' + gal + '_res7100/'
    snaps = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', snap_ids,
                                         properties=['position', 'mass', 'velocity',
                                                     'form.scalefactor'],
                                         assign_principal_axes=True,
                                         simulation_directory=sim_directory)
    for idx, snap in zip(snap_ids, snaps):
        for k in snap.keys():
            fout = k + '_' + gal + '_id' + str(idx) + '.npy'
            cyl_pos = snap[k].prop('host.distance.principal.cylindrical')

            rbool = np.logical_and(cyl_pos[:,0] > rmin, cyl_pos[:,0] < rmax)
            zbool = np.abs(cyl_pos[:,1]) < zmax
            keys = np.where(np.logical_and(rbool, zbool))[0]
            np.save(fout, cyl_pos[keys])
