#
# Emre Alca
# University of Pennsylvania
# Created on Tue Jun 23 2026
# Last Modified: 2026/09/17 17:11:55
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
# exp_dir = f'{ceph}multi_aster_disk/inital_experiments/fixed_disk_5.0um_disk_2_aster_20.0_lbar_both_10000_tubulin__0.0001_dt_0.0005_temperature_0.032_learning_rate'

num = 2
force = 'both'

# num_asters = [2]
# forces = ['push'] #['pull','push','both']
# costs = ['10000_tubulin_0.0005_temperature'] # ['no_cost', '10000_tubulin_0.0005_temperature']
# for num in num_asters:
#     for force in forces:
#         for cost in costs:

# exp_dir = f'{ceph}multi_aster/for-spindle-group-meeting/{num}_aster_{force}_{cost}'
print(f'\n\nFor {exp_dir}:\n')

spindle = mas.retrieve_experiement(experiment_dir=exp_dir, save_trajectory=True)

# -- steady state quantities --
# read the simulated duration off the recovered trajectory rather than guessing a max_time
# (spindle.time is reconstructed from the last trajectory key by insertion order, which can
# disagree with the actual max recorded time -- take the max of the saved keys directly instead)
max_time = max(spindle.trajectory.keys())
end_time = max_time
start_time = max(end_time - 100, 0)


spindle.plot_cost(end_time=end_time)
spindle.plot_aster_distance_from_centre(end_time=end_time)
if num == 2:
    spindle.plot_aster_separation(end_time=end_time)
spindle.plot_disk_distance_from_centre(end_time=end_time)
spindle.plot_disk_orientation(end_time=end_time)
spindle.plot_num_mts_per_mtoc(force=force, end_time=end_time, ylim=(0,260))
spindle.plot_num_disk_mts_per_mtoc(force=force, end_time=end_time)
spindle.plot_pull_to_push_ratio()


# occupancy by angle
spindle.plot_occupancy_vs_angle(1, start_time, end_time, show_occupancy=force)

# surface occupancy
if force == 'both':
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy='pull')
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy='push')
else:
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy=force)

# lifetime vs angle (like plot_occupancy_vs_angle)
spindle.plot_lifetime_vs_angle(1, start_time, end_time, show_occupancy=force)

# spatial lifetime map (like plot_surface_occupancy)
if force == 'both':
    spindle.plot_lifetime_surface(1, start_time, end_time, show_occupancy='pull')
    spindle.plot_lifetime_surface(1, start_time, end_time, show_occupancy='push')
else:
    spindle.plot_lifetime_surface(1, start_time, end_time, show_occupancy=force)

print(f'disk centre: {spindle.disk_center}  (target: origin)')
print(f'disk e1 (normal): {spindle.disk_e1}  (target: +x)')
num_disk_occupied = (np.count_nonzero(spindle.disk_push_state_front) + np.count_nonzero(spindle.disk_push_state_back)
                      + np.count_nonzero(spindle.disk_pull_state_front) + np.count_nonzero(spindle.disk_pull_state_back))
print(f'disk sites occupied: {num_disk_occupied} / {2 * (spindle.num_disk_push_sites + spindle.num_disk_pull_sites)}')

# plot single animation frames
spindle.plot_mtoc_trajectory_frame(5, 5, show_occupancy=True)
spindle.plot_mtoc_trajectory_frame(max_time, 5, show_occupancy=True)

ani_path = os.path.join(spindle.plot_folder_path, f'occupancy_ani.mp4')
spindle.animate_mtoc_trajectory(save_path=ani_path, interval=50, stride=100, show_occupancy=force, occupancy_mtoc_id=1, start_time=0, end_time=max_time)