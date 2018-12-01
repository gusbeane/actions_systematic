from oceanic.analysis import snapshot_action_calculator
from oceanic.options import options_reader
import dill

opt = options_reader('../options_m12i')
ac = snapshot_action_calculator(opt, snapshot_file='cluster_snapshots_m12i.p')
ac.all_actions(fileout='cluster_snapshots_m12i_actions.p')

opt = options_reader('../options_m12i_slow')
ac = snapshot_action_calculator(opt, snapshot_file='cluster_snapshots_m12i_slow.p')
ac.all_actions(fileout='cluster_snapshots_m12i_slow_actions.p')

opt = options_reader('../options_m12i_fast')
ac = snapshot_action_calculator(opt, snapshot_file='cluster_snapshots_m12i_fast.p')
ac.all_actions(fileout='cluster_snapshots_m12i_fast_actions.p')
