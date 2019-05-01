import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
import agama

def rejection_sample(p, xmin=0, xmax=1, n=10000):
    samples = []

    xlist = np.linspace(xmin, xmax, 10000)
    px = p(xlist)
    k = np.max(px)

    for i in range(n):
        z = np.random.uniform(xmin, xmax)
        pz = p(z)
        r = np.random.uniform(0, 1)
        if r < pz/k:
            samples.append(z)

    return samples

def construct_mw():
    disk = dict(type='MiyamotoNagai', mass=6.8E10, scaleRadius=3., scaleHeight=0.28)

    # Dehnen with gamma=1 is equivalent to Hernquist
    bulge = dict(type='Dehnen', gamma=1, mass=5E9, scaleRadius=1.)
    nucleus = dict(type='Dehnen', gamma=1, mass=1.71E9, scaleRadius=0.07)

    # halo
    halo = dict(type='NFW', mass=5.4E11, scaleRadius=15.62)

    pot = agama.Potential(disk, bulge, nucleus, halo)
    den = agama.Density(disk, bulge, nucleus, halo)
    return pot, den

def construct_thin_df(pot):
    Sigma0 = 36.42 * 1E6 # Msol / kpc^2
    Rd = 2.4
    sigmar0 = 27
    sigmaz0 = 20
    hdisk = 0.36
    df = agama.DistributionFunction(type='QuasiIsothermal', potential=pot, Sigma0=Sigma0,
                                    Rdisk=Rd, Rsigmar=Rd, sigmar0=sigmar0, sigmaz0=sigmaz0, Hdisk=hdisk)
    gal = agama.GalaxyModel(pot, df)

    return df, gal

def construct_thick_df(pot):
    Sigma0 = 4.05 * 1E6 # Msol / kpc^2
    Rd = 2.4
    Rdp = 2.5
    sigmar0 = 48
    sigmaz0 = 44
    hdisk = 1
    
    df = agama.DistributionFunction(type='QuasiIsothermal', potential=pot, Sigma0=Sigma0,
                                    Rdisk=Rd, Rsigmar=Rdp, sigmar0=sigmar0, sigmaz0=sigmaz0, Hdisk=hdisk)
    gal = agama.GalaxyModel(pot, df)

    return df, gal

def sample(gal, Rmin, Rmax, zmax, n, nchunk=1000000):
    samples = []

    while len(samples) < n:
        posvel, m = gal.sample(nchunk)

        R = np.linalg.norm(posvel[:,:2], axis=1)
        z = np.abs(posvel[:,2])
        Rminbool = np.greater(R, Rmin)
        Rmaxbool = np.less(R, Rmax)
        zbool = np.less(z, zmax)

        totbool = np.logical_and(np.logical_and(Rminbool, Rmaxbool), zbool)
        keys = np.where(totbool)[0]

        if len(keys) > 0:
            print('key length:', len(keys))
            if len(samples) == 0:
                samples = posvel[keys]
            else:
                samples = np.append(samples, posvel[keys], axis=0)

        # print(np.array(samples))
        print('sample length:', len(samples))

    return np.array(samples)

agama.setUnits(mass=1, length=1, velocity=1)
mw, den = construct_mw()

thin_df, thin_gal = construct_thin_df(mw)
thick_df, thick_gal = construct_thick_df(mw)


