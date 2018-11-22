from oceanic.analysis import snapshot_action_calculator
from oceanic.options import options_reader
import dill

opt = options_reader('../options')
#ac = snapshot_action_calculator(opt)
ac = snapshot_action_calculator(opt, snapshot_file='cluster_snapshots.p')

ac.all_actions(fileout='cluster_snapshots_actions.p')
