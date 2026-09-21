#
# Emre Alca
# University of Pennsylvania
# Created on Tue Jun 09 2026
# Last Modified: 2026/08/16 15:39:50
#


import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import argparse
import os


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

num_attempts = 20000000

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# making push/pull lattices with 10 um radii
mesh_density = 8
num_motors = 100
cell_radius = 20 # um
tubulin_budget = 10000 #num_mts*cell_radius
num_mts = tubulin_budget / cell_radius
optimization_temperature = 1e-3

growth_rate = 0.5 # um / s
catastrophe_rate = 0.025 # 1 / s
average_mt_length = growth_rate / catastrophe_rate # um

learning_rate = np.round(1 /  ((tubulin_budget / 0.4) * np.square(catastrophe_rate) / growth_rate ), 5)
# learning_rate = 0.01

spindle_length = 20 # um

print(f'learning rate: {learning_rate} s')

# order_of_tubulin = 4 # 10^n um tubulin
# tubulin_budget = 10**order_of_tubulin
# dt = 10 ** (-(order_of_tubulin - 2))

# euler_dt = 1e-3
# if dt < 1e-3:
#     euler_dt = dt / 10
    

trimesh = np.load(f'{home}multi-aster/trimesh_cache/sphere_{mesh_density}_subdivs_1_radius.npy') * cell_radius
sloan_100 = np.load(f'{home}multi-aster/sloane_cache/sloane_{num_motors}.npy') * cell_radius

print(trimesh.shape)

# try centring a single aster
mtoc_positions = np.array([[0,0,1], [0,0,-1]])
# mtoc_positions = np.array([[0,0,5]])

force='both'

spindle = mas.Spindle(initial_mtoc_positions=mtoc_positions, 
            push_lattice=trimesh,
            pull_lattice=sloan_100,
            boundary_radius=cell_radius,
            tubulin_budget=tubulin_budget,
            growth_rate=growth_rate,
            spindle_length=spindle_length,
            stall_force=10.0,
            rigidity=5,
            pull_force=10.0,
            average_mt_length=average_mt_length,
            optimization_temperature=optimization_temperature,
            evolution_time=learning_rate,
            euler_timestep_size=learning_rate / 10,
            data_dir=f'{ceph}multi_aster/for-spindle-group-meeting/',
            # dir_prefix=f'{len(mtoc_positions)}_aster_{force}_no_cost',
            dir_prefix=f'{len(mtoc_positions)}_aster_{average_mt_length}_lbar_{force}_{tubulin_budget}_tubulin_{spindle_length}_sep_{1e-4}_dt_{optimization_temperature}_temperature_{learning_rate}_learning_rate',
            save_trajectory=True,
            save=True
            )

# num_pulling = 0
# num_pushing = 0
# prepop_trace = []
# for i in range(int(.6 * 100)):
#     push, lattice_site, site_value = spindle.sample_spindle_update(mtoc_id=1, add=True)

#     if push:
#         old_site_value = spindle.push_state[lattice_site]
#         spindle.push_state[lattice_site] = site_value
#         num_pushing += 1
#     else:
#         old_site_value = spindle.pull_state[lattice_site]
#         spindle.pull_state[lattice_site] = site_value
#         num_pulling += 1

#     prepop_trace.append((push, lattice_site, site_value))

# # record the pre-population as an initial spindle_trace batch so that
# # calculate_num_mts_per_mtoc_over_time / calculate_motor_occupancy / calculate_push_occupancy
# # (which reconstruct state by replaying spindle_trace from zero) see these MTs too
# if spindle.save and prepop_trace:
#     np.save(
#         os.path.join(spindle.spindle_trace_path, f'spindle_trace_0_{len(prepop_trace)}.npy'),
#         np.array(prepop_trace)
#     )
#     spindle.num_accepted_states = len(prepop_trace)

# print(num_pulling)
# print(num_pushing)

# print(len(np.where(spindle.pull_state==1)[0]), len(np.where(spindle.push_state==1)[0]))

# print(spindle.calc_mtoc_velocity(1), spindle.calc_mtoc_velocity(2))

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
spindle.animate_mtoc_trajectory(save_path=ani_path, interval=50, stride=100, show_occupancy=force, occupancy_mtoc_id=1)
