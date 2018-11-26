import numpy as np
import gizmo_analysis as gizmo
from tqdm import tqdm

start = 590
end = 600 
snap_ids = list(range(start, end+1))

pa = np.array([[-0.116813976049, 0.981662058456, -0.150645603947],
               [-0.860269343337, -0.024217138031, 0.509264358796],
               [-0.496277293369, -0.189084989395, -0.847322674588]])

rmin = 7.2
rmax = 9.2
zmax = 1.0

#for gal in ['m12i', 'm12f']:
for gal in ['m12i']:
    sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/' + gal + '_res7100/'
    snaps = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', snap_ids,
                                         properties=['position', 'mass', 'velocity',
                                                     'form.scalefactor'],
                                         assign_principal_axes=True,
                                         simulation_directory=sim_directory)
    for idx, snap in tqdm(zip(snap_ids, snaps)):
        #snap.principal_axes_vectors = snaps[0].principal_axes_vectors
        snap.principal_axes_vectors = pa 
        for k in snap.keys():
            snap[k].principal_axes_vectors = pa
            fout = k + '_' + gal + '_id' + str(idx) + '.npy'
            cyl_pos = snap[k].prop('host.distance.principal.cylindrical')

            rbool = np.logical_and(cyl_pos[:,0] > rmin, cyl_pos[:,0] < rmax)
            zbool = np.abs(cyl_pos[:,1]) < zmax
            keys = np.where(np.logical_and(rbool, zbool))[0]
            np.save(fout, cyl_pos[keys])
