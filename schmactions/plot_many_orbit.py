import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp

from scipy.stats import sigmaclip

import warnings

from tqdm import tqdm
import sys
import pickle

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

ncpu = int(sys.argv[1])

wrong_max = 1000 # times dt, so 1 Gyr
max_offset = 500
d_offset = 50

JrIQR_max = 100
LzIQR_max = 10
JzIQR_max = 200

mw = gp.MilkyWayPotential()

def compute_actions(pos=None, vel=None, phase=None):
    if phase is None:
        q = gd.PhaseSpacePosition(pos=pos, vel=vel)
    else:
        q = phase
    orbit = mw.integrate_orbit(q, dt=dt, t1=t1, t2=t2, Integrator=gi.DOPRI853Integrator)
    res = gd.find_actions(orbit, N_max=8)
    ans = res['actions'].to(u.kpc * u.km / u.s).value
    return ans


def compute_actions_wrong_ref_frame(init_pos, init_vel, offset, cadence=25, wrong_max=None):
    q = gd.PhaseSpacePosition(pos=init_pos, vel=init_vel)
    orbit = mw.integrate_orbit(q, dt=dt, t1=t1, t2=t2, Integrator=gi.DOPRI853Integrator)

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
            try:
                this_orbit = mw.integrate_orbit(this_q, dt=dt, t1=t1, t2=t2, Integrator=gi.DOPRI853Integrator)
                res = gd.actionangle.find_actions(this_orbit, N_max=8)
                ans = res['actions'].to(u.kpc * u.km / u.s).value
            except:
                ans = [np.nan, np.nan, np.nan]
            out_action.append(ans)
    #out_action = Parallel(n_jobs=ncpu) (delayed(loop)(this_q) for this_q in tqdm(orbit[:wrong_max][::cadence]))
    return time, np.array(out_action), orbit

def init_fig():
    fig, ax = plt.subplots(2, 3, figsize=(7, 5))
    for x in ax[0]:
        x.set_xlabel(r'$z\,\text{offset}\,[\,\text{pc}\,]$')
        x.set_xlim(0, max_offset)
        x.set_xticks(np.arange(0, max_offset, 50), minor=True)
    for x in ax[1]:
        x.set_xlabel(r'$R\,\text{offset}\,[\,\text{pc}\,]$')
        x.set_xlim(0, max_offset)
        x.set_xticks(np.arange(0, max_offset, 50), minor=True)

    for x in ax[:,0]:
        x.set_ylabel(r'$\Delta J_r / J_{r,\text{true}}\,[\,\%\,]$')
        x.set_ylim(0, JrIQR_max)
    for x in ax[:,1]:
        x.set_ylabel(r'$\Delta L_z / L_{z,\text{true}}\,[\,\%\,]$')
        x.set_ylim(0, LzIQR_max)
    for x in ax[:,2]:
        x.set_ylabel(r'$\Delta J_z / J_{z,\text{true}}\,[\,\%\,]$')
        x.set_ylim(0, JzIQR_max)

    return fig, ax

def save_fig(fig, ax, out, true_act=False):
    fig.tight_layout()
    if true_act:
        for x in ax[0]:
            x.legend(frameon=False)
    fig.savefig(out)

def plot_wrong_act(fig, ax, off, perc, c=tb_c[0], true_act=None):
    # perform a 5 sigma clip to remove numerical artifacts
    _, pJr_low, pJr_high = sigmaclip(perc[:,0], low=5, high=5)
    _, pLz_low, pLz_high = sigmaclip(perc[:,1], low=5, high=5)
    _, pJz_low, pJz_high = sigmaclip(perc[:,2], low=5, high=5)
    pJrbool = np.logical_and(perc[:,0] > pJr_low, perc[:,0] < pJr_high)
    pLzbool = np.logical_and(perc[:,1] > pLz_low, perc[:,1] < pLz_high)
    pJzbool = np.logical_and(perc[:,2] > pJz_low, perc[:,2] < pJz_high)
    keys = np.where(np.logical_and(np.logical_and(pJrbool, pLzbool), pJzbool))[0]

    off = off[keys]
    perc = perc[keys]
    
    pJrmed = np.full(len(perc), np.median(perc[:,0]))
    pLzmed = np.full(len(perc), np.median(perc[:,1]))
    pJzmed = np.full(len(perc), np.median(perc[:,2]))

    y1 = 100 * perc[:,0]/true_act[0]
    y2 = 100 * perc[:,1]/np.abs(true_act[1])
    y3 = 100 * perc[:,2]/true_act[2]
    
    if true_act is not None:
        label0 = "{0:0.1f}".format(true_act[0])
        label1 = "{0:0.1f}".format(true_act[1])
        label2 = "{0:0.1f}".format(true_act[2])
    else:
        label0 = None
        label1 = None
        label2 = None

    ax[0].plot(off, y1, c=c, label=label0)
    ax[1].plot(off, y2, c=c, label=label1)
    ax[2].plot(off, y3, c=c, label=label2)


    return fig, ax

def z_off(num, z=False, R=False, init_pos=None, init_vel=None):
    # some fiducial position
    if init_pos is None:
        init_pos = [8, 0, 0] * u.kpc
    
    # some fiducial velocity
    if init_vel is None:
        init_vel = [0, -190, 30] * u.km/u.s

    no_offset = [0, 0, 0] * u.kpc
    t, correct_act, orbit = compute_actions_wrong_ref_frame(init_pos, init_vel, no_offset, cadence=10, wrong_max=wrong_max)
    
    if z:
        this_offset = [0, 0, num] * u.pc
    elif R:
        this_offset = [num, 0, 0] * u.pc
    else:
        return None
    t, act, orbit = compute_actions_wrong_ref_frame(init_pos, init_vel, this_offset, cadence=10, wrong_max=wrong_max)
    return np.percentile(act, 95, axis=0) - np.percentile(act, 5, axis=0)

def run_offlist(z=False, R=False, init_pos=None, init_vel=None):
    if z:
        xlabel = r'$z\,\text{offset}\,[\,\text{pc}\,]$'
    elif R:
        xlabel = r'$R\,\text{offset}\,[\,\text{pc}\,]$'
    else:
        return None

    offlist = np.linspace(0, max_offset, d_offset) # u.pc
    perc_list = Parallel(n_jobs=ncpu) (delayed(z_off)(num, z, R, init_pos, init_vel) for num in tqdm(offlist))
    perc_list = np.array(perc_list)
    return offlist, perc_list

if __name__ == '__main__':
    try:
        result_z = pickle.load(open('result_z.p', 'rb'))
        result_R = pickle.load(open('result_R.p', 'rb'))
        init_act_list = pickle.load(open('init_act_list.p', 'rb'))
    except:
        init_pos = [8, 0, 0] * u.kpc
        init_vel_list = [[0, -190, 10] * u.km/u.s,
                         [0, -190, 30] * u.km/u.s,
                         [0, -190, 50] * u.km/u.s]
        init_act_list = [compute_actions(init_pos, init_vel) for init_vel in init_vel_list]

        result_z = [run_offlist(z=True, init_pos=init_pos, init_vel=init_vel) for init_vel in init_vel_list]
        result_R = [run_offlist(R=True, init_pos=init_pos, init_vel=init_vel) for init_vel in init_vel_list]

        pickle.dump(result_z, open('result_z.p', 'wb'))
        pickle.dump(result_R, open('result_R.p', 'wb'))
        pickle.dump(init_act_list, open('init_act_list.p', 'wb'))

    fig, ax = init_fig()
    for r_z, r_R, act, c in zip(result_z, result_R, init_act_list, tb_c):
        offlist = r_z[0]
        perc_list = r_z[1]
        fig, ax[0] = plot_wrong_act(fig, ax[0], offlist, perc_list, c=c, true_act = act)
        
        offlist = r_R[0]
        perc_list = r_R[1]
        fig, ax[1] = plot_wrong_act(fig, ax[1], offlist, perc_list, c=c, true_act = act)

    save_fig(fig, ax, 'test.pdf', true_act=True)


