import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp

from scipy.stats import sigmaclip

import warnings

from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

dt = 1 * u.Myr
t1 = 0.0 * u.Gyr
t2 = 5.0 * u.Gyr

#ncpu = 8
ncpu = 40

wrong_max = 1000 # times dt, so 1 Gyr

mw = gp.MilkyWayPotential()
def compute_actions_wrong_ref_frame(init_pos, init_vel, offset, cadence=25, wrong_max=None):
    q = gd.PhaseSpacePosition(pos=init_pos, vel=init_vel)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        orbit = mw.integrate_orbit(q, dt=dt, t1=t1, t2=t2, Integrator=gi.DOPRI853Integrator)
        res = gd.find_actions(orbit, N_max=8)
        ans = res['actions'].to(u.kpc * u.km / u.s).value
        print(ans)

    pos = np.transpose(orbit.pos.xyz.to_value(u.kpc))
    offset = offset.to_value(u.kpc)
    pos = np.subtract(pos, offset)
    vel = orbit.vel.d_xyz
    pos = np.transpose(pos) * u.kpc
    off_q = gd.PhaseSpacePosition(pos=pos, vel=vel)

    if wrong_max is None:
        wrong_max = len(orbit)

    out_action = []
    time = orbit.t[:wrong_max][::cadence]
    
    for this_q in tqdm(off_q[:wrong_max][::cadence]):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            this_orbit = mw.integrate_orbit(this_q, dt=dt, t1=t1, t2=t2, Integrator=gi.DOPRI853Integrator)
            res = gd.actionangle.find_actions(this_orbit, N_max=8)
            ans = res['actions'].to(u.kpc * u.km / u.s).value
            out_action.append(ans)
    #out_action = Parallel(n_jobs=ncpu) (delayed(loop)(this_q) for this_q in tqdm(orbit[:wrong_max][::cadence]))
    return time, np.array(out_action), orbit

def plot_wrong_act(t, act, rel_error=False, many_orbits=False, x_label=None):
    fig, ax = plt.subplots(1, 3, figsize=(7, 2.5))
    
    # perform a 5 sigma clip to remove numerical artifacts
    _, Jr_low, Jr_high = sigmaclip(act[:,0], low=5, high=5)
    _, Lz_low, Lz_high = sigmaclip(act[:,1], low=5, high=5)
    _, Jz_low, Jz_high = sigmaclip(act[:,2], low=5, high=5)
    Jrbool = np.logical_and(act[:,0] > Jr_low, act[:,0] < Jr_high)
    Lzbool = np.logical_and(act[:,1] > Lz_low, act[:,1] < Lz_high)
    Jzbool = np.logical_and(act[:,2] > Jz_low, act[:,2] < Jz_high)
    keys = np.where(np.logical_and(np.logical_and(Jrbool, Lzbool), Jzbool))[0]
    print(len(keys), len(act[:,0]))

    t = t[keys]
    act = act[keys]
    
    Jrmed = np.full(len(act), np.median(act[:,0]))
    Lzmed = np.full(len(act), np.median(act[:,1]))
    Jzmed = np.full(len(act), np.median(act[:,2]))

    if rel_error:
        y1 = (act[:,0] - Jrmed)/act[:,0] 
        y2 = (act[:,1] - Lzmed)/act[:,1] 
        y3 = (act[:,2] - Jzmed)/act[:,2] 
    else:
        y1 = act[:,0]
        y2 = act[:,1]
        y3 = act[:,2]
    ax[0].plot(t, y1, c=tb_c[0])
    ax[1].plot(t, y2, c=tb_c[0])
    ax[2].plot(t, y3, c=tb_c[0])

    if not rel_error and not many_orbits:
        ax[0].plot(t, Jrmed, c=tb_c[0], ls='dashed')
        ax[1].plot(t, Lzmed, c=tb_c[0], ls='dashed')
        ax[2].plot(t, Jzmed, c=tb_c[0], ls='dashed')

    ax[0].set_xlim(0, wrong_max)
    ax[1].set_xlim(0, wrong_max)
    ax[2].set_xlim(0, wrong_max)
    
    y_formatter = mpl.ticker.ScalarFormatter(useOffset=True)
    for x in ax:
        if x_label is None:
            x.set_xlabel(r'$t\,[\,\text{Myr}\,]$')
        else:
            x.set_xlabel(x_label)
        if rel_error:
            #x.yaxis.set_major_formatter(y_formatter)
            x.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if rel_error:
        ax[0].set_ylabel(r'$J_r\,\text{error}$')
        ax[1].set_ylabel(r'$L_z\,\text{error}$')
        ax[2].set_ylabel(r'$J_z\,\text{error}$')
    elif not many_orbits:
        ax[0].set_ylabel(r'$J_r\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
        ax[1].set_ylabel(r'$L_z\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
        ax[2].set_ylabel(r'$J_z\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
    else:
        ax[0].set_ylabel(r'$J_r\,\text{IQR}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
        ax[1].set_ylabel(r'$L_z\,\text{IQR}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
        ax[2].set_ylabel(r'$J_z\,\text{IQR}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
        
    fig.tight_layout()
    return fig, ax


def run_one_orbit(z=False, R=False):
    init_pos = [8, 0, 0] * u.kpc
    init_vel = [0, -190, 30] * u.km/u.s
    if z:
        offset = [0, 0, 100] * u.pc
    elif R:
        offset = [100, 0, 0] * u.pc
    else:
        return None

    t, act, orbit = compute_actions_wrong_ref_frame(init_pos, init_vel, offset, cadence=1, wrong_max=wrong_max)
    plot_wrong_act(t, act)
    if z:
        out = 'one_orbit_z.pdf'
    elif R:
        out = 'one_orbit_R.pdf'
    plt.savefig(out)
    plt.close()

    perc = np.percentile(act, 95, axis=0) - np.percentile(act, 5, axis=0)
    frac = perc/np.median(act, axis=0)
    print('95th minus 5th percentiles:', perc)
    print('fractional error:', frac)

def run_numerical_orbit():
    offset = [0, 0, 0] * u.pc
    t_n, act_n, orbit_n = compute_actions_wrong_ref_frame(init_pos, init_vel, offset, cadence=1, wrong_max=wrong_max)
    plot_wrong_act(t_n, act_n, rel_error=True)
    plt.savefig('one_orbit_numerical_check.pdf')
    plt.close()

def z_off(num, z=False, R=False):
    init_pos = [8, 0, 0] * u.kpc
    init_vel = [0, -190, 30] * u.km/u.s
    if z:
        this_offset = [0, 0, num] * u.pc
    elif R:
        this_offset = [num, 0, 0] * u.pc
    else:
        return None
    t, act, orbit = compute_actions_wrong_ref_frame(init_pos, init_vel, this_offset, cadence=10, wrong_max=wrong_max)
    return np.percentile(act, 95, axis=0) - np.percentile(act, 5, axis=0)

def run_offlist(z=False, R=False):
    if z:
        xlabel = r'$z\,\text{offset}\,[\,\text{pc}\,]$'
    elif R:
        xlabel = r'$R\,\text{offset}\,[\,\text{pc}\,]$'
    else:
        return None

    offlist = np.linspace(0, 500, 50) # u.pc
    perc_list = Parallel(n_jobs=ncpu) (delayed(z_off)(num, z, R) for num in tqdm(offlist))
    perc_list = np.array(perc_list)
    fig, ax = plot_wrong_act(offlist, perc_list, rel_error=False, many_orbits=True, x_label=xlabel)
    if z:
        fig.savefig('offset_z.pdf')
    elif R:
        fig.savefig('offset_R.pdf')


if __name__ == '__main__':
    #run_one_orbit(z=True)
    #run_one_orbit(R=True)
    #run_numerical_orbit()
    run_offlist(z=True)
    #run_offlist(R=True)

