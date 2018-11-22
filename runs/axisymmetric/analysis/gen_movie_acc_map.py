from oceanic.gizmo_interface import gizmo_interface
from oceanic.options import options_reader
from oceanic.analysis import cluster_animator 
from oceanic.analysis import acceleration_heatmap
from oceanic.oceanic_io import load_interface
import sys
import dill

options_file = sys.argv[1]
cluster_file = sys.argv[2]

cluster = dill.load(open(cluster_file, 'rb'))

xcenter = 0.0
ycenter = 0.0
xmin = -0.1
xmax = 0.1
ymin = -0.1
ymax = 0.1
nres = 360
zval = 0.0

xaxis='x'
yaxis='y'
acc='x'

start=None
end=None

cmap = 'bwr_r'
cmin = -0.5
cmax = 0.5

# pLz_bound = 4.0
# pJr_bound = 2.0
# pJz_bound = 1.5
pLz_bound = 20.0
pJr_bound = 5.0
pJz_bound = 5.0
normalize=False

color_by_dist = True
log_distance = True
dist_vmin = -1
dist_vmax = 2

axisymmetric = True

plot_panel=True
acc_map=False

opt = options_reader(options_file)

#interface = load_interface('../interface', skinny=True)
interface = None

def run():
    an = cluster_animator(cluster, xaxis=xaxis, yaxis=yaxis, xmin=xmin, xmax=xmax,
                ymin=ymin, ymax=ymax, start=start, end=end, acc_map=acc_map,
                nres=nres, acc=acc, cmap=cmap, cmin=cmin, cmax=cmax, interface=interface,
                options=options_file, plot_panel=plot_panel, pLz_bound=pLz_bound,
                pJr_bound=pJr_bound, pJz_bound=pJz_bound, normalize=normalize,
                color_by_dist=color_by_dist, log_distance=log_distance,
                dist_vmin=dist_vmin, dist_vmax=dist_vmax, axisymmetric=axisymmetric)
    an()

if len(sys.argv) == 4:
    key = int(sys.argv[3])
    time = cluster[key]['time']
    ac = acceleration_heatmap(options_file, interface)
    ac(time, index=key, cache=True, return_heatmap=True,
        xcenter=xcenter, ycenter=ycenter, plot_xmin=xmin,
        plot_xmax = xmax, plot_ymin = ymin, plot_ymax=ymax,
        nres=nres, zval=zval)
else:
    #run()
    xmin = -0.3
    xmax = 0.3
    ymin = -0.3
    ymax = 0.3
    pLz_bound = 40.0
    pJr_bound = 30.0
    pJz_bound = 30.0
    log_distance = False
    dist_vmin = 0
    dist_vmax = 300
    run()
