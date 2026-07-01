#
# Emre Alca
# University of Pennsylvania
# Created on Tue Jun 09 2026
# Last Modified: 2026/06/15 15:07:12
#


import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import argparse


# --- set basic numbers from argparse ---
# parser = argparse.ArgumentParser()
# parser.add_argument("--evolution_time", type=float, default=0.0001)
# parser.add_argument("--optimization_temperature", type=float, default=0.0001)

# args = parser.parse_args()
# evolution_time = args.evolution_time
# optimization_temperature = args.optimization_temperature
# timestep_size = evolution_time / 10

# if timestep_size > 0.001:
#     timestep_size = 0.001

num_attempts = 1000000

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# making push/pull lattices with 10 um radii
mesh_density = 6
num_motors = 100
cell_radius = 10 # um
num_mts = 1000
tubulin_budget = num_mts*cell_radius
optimization_temperature = 1e-4

trimesh = np.load(f'{home}multi-aster/trimesh_cache/sphere_{mesh_density}_subdivs_1_radius.npy') * cell_radius
sloan_100 = np.load(f'{home}multi-aster/sloane_cache/sloane_{num_motors}.npy') * cell_radius

# try centring a single aster
mtoc_positions = np.array([[0,0,0], [0,0,0]])

spindle = mas.Spindle(initial_mtoc_positions=mtoc_positions, 
            push_lattice=trimesh, 
            pull_lattice=sloan_100, 
            tubulin_budget=tubulin_budget,
            stall_force=0.0,
            # evolution_time=0.5,
            # euler_timestep_size=0.001,
            optimization_temperature=optimization_temperature,
            data_dir=f'{ceph}multi_aster/dt-L',
            dir_prefix=f'2_aster_test_pushing_and_pulling_{tubulin_budget}_tubulin_{optimization_temperature}_temp',
            save_trajectory=False,
            save=True
            )

spindle.optimize(num_attempts)
# spindle.plot_cost()
