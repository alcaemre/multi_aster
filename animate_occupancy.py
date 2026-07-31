#
# Emre Alca
# University of Pennsylvania
# Created on Tue Jun 23 2026
# Last Modified: 2026/07/31 14:39:08
#

import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

exp_dir = f'{ceph}multi_aster/for-spindle-group-meeting/2_aster_2.0_lbar_both_4000_tubulin_0.005_temperature'

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
end_time = np.round(spindle.time)

end_time = 300
start_time = end_time - 100


spindle.plot_cost(end_time=end_time)
spindle.plot_aster_distance_from_centre(end_time=end_time)
if num == 2:
    spindle.plot_aster_separation(end_time=end_time)
spindle.plot_num_mts_per_mtoc(force=force, end_time=end_time, ylim=(0,260))
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

# plot single animation frames
spindle.plot_mtoc_trajectory_frame(5, 5, show_occupancy=True)
spindle.plot_mtoc_trajectory_frame(300, 5, show_occupancy=True)

# ani_path = os.path.join(spindle.plot_folder_path, f'occupancy_ani.mp4')
# spindle.animate_mtoc_trajectory(save_path=ani_path, interval=50, stride=1000, show_occupancy=force, occupancy_mtoc_id=1)