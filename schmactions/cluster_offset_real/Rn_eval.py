import pickle
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

mlist = np.array([ 352, 410, 112, 400, np.nan, np.nan, np.nan, 800, 550])
t = Table.read('real_cluster_gc.fits', format='fits')
name_list = t['cluster']

gal_list = ['m12i', 'm12f', 'm12m']

for name, mass in zip(name_list, mlist):
    for gal in gal_list:
        t = pickle.load(open('Rn_vs_mc_'+gal+'_'+name+'.p', 'rb'))
        mlist = t[:,0]
        Rn = t[:,1]
        i = interp1d(mlist, Rn)
        if np.isnan(mass):
            this_Rn = np.nan
        else:
            this_Rn = i(mass)
        print(name, gal, this_Rn*1000)


