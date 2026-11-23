#
# Emre Alca
# University of Pennsylvania
# Created on Tue Jun 23 2026
# Last Modified: 2026/11/23 09:40:21
#
import numpy as np

import spindle as mas

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os
import traceback

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# exp_dir = f'{ceph}multi_aster_disk/one_lattice/prefactors_0_0.1_0_0_1_1_0_0_2_aster_20.0_lbar_both_30000_tubulin_0.0008_temperature_0.016_learning_rate'
# exp_dir = f'{ceph}multi_aster_disk/one_lattice/prefactors_0_0.01_0_0_1_1_0_0_2_aster_20.0_lbar_both_30000_tubulin_0.0008_temperature_0.016_learning_rate'
# exp_dir = f'{ceph}multi_aster_disk/one_lattice/prefactors_0_1_0_0_1_1_0_0_2_aster_20.0_lbar_both_30000_tubulin_0.0008_temperature_0.016_learning_rate'
exp_dir = f'{ceph}against-baseline/cat_0.05_prefactors_50_1_1_0_100_100_0_0_0_2_aster_10.0_lbar_both_10000_tubulin_200_motors_0.0001_temperature_0.019999999999999997_learning_rate'

print(f'restarting {exp_dir}')

spindle = mas.retrieve_experiement(experiment_dir=exp_dir, save_trajectory=True, save=True)

# print(spindle.num_accepted_states)

num_attempts = 100000000
max_time = 5000

# one proposal-evaluating worker per core SLURM gives the job (see Spindle.optimize / ProposalPool)
num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

# -- plotting --
force = 'both'
plot_every = 10000 # accepted spindle states between plot refreshes
occupancy_window = 100 # s, window over which occupancy and lifetimes are averaged


def safe_make_plots(spindle):
    # a plotting failure should cost one set of plots, not a multi-hour run
    try:
        mas.make_plots(spindle, force=force, window=occupancy_window)
    except Exception:
        traceback.print_exc()
        plt.close('all')


spindle.optimize(num_attempts, max_lab_time=max_time, callback=safe_make_plots, callback_every=plot_every, num_workers=num_workers)

# final set of plots for wherever the run stopped
safe_make_plots(spindle)

# animate_mtoc_trajectory already overlays the disk (its pose, characteristic vectors e1/e2/e3,
# and site occupancy) alongside the MTOCs -- see multi_aster_spindle.py
ani_path = os.path.join(spindle.plot_folder_path, f'occupancy_ani.mp4')
spindle.animate_mtoc_trajectory(save_path=ani_path, interval=50, stride=100, show_occupancy=force, occupancy_mtoc_id=1, start_time=0, end_time=max_time)
