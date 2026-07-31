#
# Emre Alca
# University of Pennsylvania
# Created on Tue Jun 23 2026
# Last Modified: 2026/07/23 16:46:27
#
import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# exp_dir = f'{ceph}multi_aster/useful/spindle_centring_10.0_lbar_5000_tubulin_0.0005_temp'
# exp_dir = f'{ceph}multi_aster/dt-L/single_aster_positioning_2500_tubulin_0.0005_temp'
# exp_dir = f'{ceph}multi_aster/useful/single_aster_positioning_reduced_tub_cost_1000_tubulin_0.002_temp'
# exp_dir = f'{ceph}multi_aster/dt-L/two_aster_8_spindle_length_2000_tubulin_0.002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/two_aster_14_spindle_length_2000_tubulin_0.002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/sap_force_1000_tubulin_0.002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/exponential_2.0_lbar_10000_tubulin_0.002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/uniform_13_dbar_10000_tubulin_0.001_temp'
# exp_dir = f'{ceph}multi_aster/dt-L/uniform_13_dbar_10000_tubulin_0.0001_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/test_new_saving_13_dbar_10000_tubulin_0.0001_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/test_new_saving_13_dbar_10000_tubulin_0.0002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/test_new_saving_13_dbar_10000_tubulin_0.0005_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/set_length_alignment_zero_force_13_dbar_10000_tubulin_0.0005_temp/'
# exp_dir = f'{ceph}multi_aster/for-spindle-group-meeting/2_aster_both_10000_tubulin_0.0005_temperature'
exp_dir = f'{ceph}multi_aster/for-spindle-group-meeting/2_2.0_lbar_both_9_spindle_length_5e-05_temperature'

print(f'restarting {exp_dir}')

spindle = mas.retrieve_experiement(experiment_dir=exp_dir, save_trajectory=True, save=True)

# print(spindle.num_accepted_states)

spindle.optimize(5000000, max_lab_time=500)

force = 'both'

spindle.plot_cost()
spindle.plot_aster_distance_from_centre()
spindle.plot_aster_separation()
spindle.plot_num_mts_per_mtoc()

end_time = np.round(spindle.time)
start_time = end_time - 100
spindle.plot_occupancy_vs_angle(1, start_time, end_time, show_occupancy=force)
if force == 'both':
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy='pull')
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy='push')
    spindle.plot_pull_to_push_ratio()
else:
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy=force)

ani_path = os.path.join(spindle.plot_folder_path, f'occupancy_ani.mp4')
spindle.animate_mtoc_trajectory(save_path=ani_path, interval=50, stride=1000, show_occupancy=force, occupancy_mtoc_id=1)