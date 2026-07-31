#
# Emre Alca
# University of Pennsylvania
# Created on Thu Jul 09 2026
# Last Modified: 2026/07/16 13:14:51
#

import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import argparse
import os

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'
# home = '/Users/emrealca/home/'

num_attempts = 1000000

mesh_density = 6
cell_radius = 10 # um

# pull lattice
sloan_60 = np.load(f'{home}multi-aster/sloane_cache/sloane_{60}.npy') * cell_radius
sloan_60 = sloan_60[sloan_60[:,0] < 0]
sloan_120 = np.load(f'{home}multi-aster/sloane_cache/sloane_{120}.npy') * cell_radius
sloan_120 = sloan_120[sloan_120[:,0] > 0]
asymmetric_motor_lattice = np.vstack([sloan_120, sloan_60])  

# push_lattice
trimesh = np.load(f'{home}multi-aster/trimesh_cache/sphere_{mesh_density}_subdivs_1_radius.npy') * cell_radius

num_mts = 400
tubulin_budget = num_mts*cell_radius
optimization_temperature = 5e-4
average_mt_length = 2.0

mtoc_positions = np.array([[0,0,1], [0,0,-1]])
# mtoc_positions = np.array([[0,0,5]])
force='both'

spindle = mas.Spindle(initial_mtoc_positions=mtoc_positions, 
            push_lattice=trimesh, 
            pull_lattice=asymmetric_motor_lattice, 
            tubulin_budget=tubulin_budget,
            stall_force=0.0,
            # rigidity=0.0,
            pull_force=10.0,
            average_mt_length=average_mt_length,
            optimization_temperature=optimization_temperature,
            data_dir=f'{ceph}multi_aster/for-spindle-group-meeting/',
            dir_prefix=f'asym_{len(mtoc_positions)}_aster_{force}_{tubulin_budget}_tubulin_{optimization_temperature}_temperature',
            # dir_prefix=f'pull_length_alignment_zero_force_{spindle_length}_dbar_{tubulin_budget}_tubulin_{optimization_temperature}_temp',
            save_trajectory=True,
            save=True
            )

spindle.optimize(num_attempts, max_lab_time=500)

spindle.plot_cost()
spindle.plot_aster_distance_from_centre()
if len(mtoc_positions) == 2:
    spindle.plot_aster_separation()
spindle.plot_num_mts_per_mtoc()

end_time = np.round(spindle.time)
start_time = end_time - 100
spindle.plot_occupancy_vs_angle(1, start_time, end_time, show_occupancy=force)
if force == 'both':
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy='pull')
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy='push')
else:
    spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy=force)

ani_path = os.path.join(spindle.plot_folder_path, f'occupancy_ani.mp4')
spindle.animate_mtoc_trajectory(save_path=ani_path, interval=50, stride=1000, show_occupancy=force, occupancy_mtoc_id=1)
