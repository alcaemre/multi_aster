#
# Emre Alca
# University of Pennsylvania
# Created on Tue Jun 23 2026
# Last Modified: 2026/09/17 17:20:20
#
import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

exp_dir = f'{ceph}multi_aster_disk/inital_experiments/fixed_mtocs_rigid_pushers_5.0um_disk_2_aster_20.0_lbar_both_10000_tubulin__0.0001_dt_0.001_temperature_0.032_learning_rate'

print(f'restarting {exp_dir}')

spindle = mas.retrieve_experiement(experiment_dir=exp_dir, save_trajectory=True, save=True)

# print(spindle.num_accepted_states)

num_attempts = 5000000
max_time = 5000

spindle.optimize(num_attempts, max_lab_time=max_time)

force = 'pull'


spindle.plot_cost()
spindle.plot_aster_distance_from_centre()
if len(list(spindle.mtoc_positions.keys())) == 2:
    spindle.plot_aster_separation()
spindle.plot_disk_distance_from_centre()
spindle.plot_disk_orientation()
spindle.plot_num_mts_per_mtoc()
spindle.plot_num_disk_mts_per_mtoc()

end_time = np.round(spindle.time)
start_time = end_time - 100
spindle.plot_occupancy_vs_angle(1, start_time, end_time, show_occupancy=force)
if force == 'both':
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy='pull')
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy='push')
else:
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy=force)

print(f'disk centre: {spindle.disk_center}  (target: origin)')
print(f'disk e1 (normal): {spindle.disk_e1}  (target: +x)')
num_disk_occupied = (np.count_nonzero(spindle.disk_push_state_front) + np.count_nonzero(spindle.disk_push_state_back)
                      + np.count_nonzero(spindle.disk_pull_state_front) + np.count_nonzero(spindle.disk_pull_state_back))
# print(f'disk sites occupied: {num_disk_occupied} / {2 * (num_disk_push_sites + num_disk_pull_sites)}')

# animate_mtoc_trajectory already overlays the disk (its pose, characteristic vectors e1/e2/e3,
# and site occupancy) alongside the MTOCs -- see multi_aster_spindle.py
ani_path = os.path.join(spindle.plot_folder_path, f'occupancy_ani.mp4')
spindle.animate_mtoc_trajectory(save_path=ani_path, interval=50, stride=100, show_occupancy=force, occupancy_mtoc_id=1, start_time=0, end_time=max_time)
