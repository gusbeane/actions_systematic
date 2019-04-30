import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

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

