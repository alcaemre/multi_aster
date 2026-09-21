#
# Emre Alca
# University of Pennsylvania
# Created on Thu Aug 20 2026
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

num_attempts = 10000000
max_time = 500

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# making push/pull lattices with 10 um radii
# kept smaller than two-aster-centring.py's mesh_density=8 (655k points) -- with that many
# boundary sites, the disk's own sites (a few hundred at most) would almost never win the
# weighted draw in sample_spindle_update purely on numbers, regardless of relative distance.
mesh_density = 7
num_motors = 100
cell_radius = 20 # um
tubulin_budget = 10000 #num_mts*cell_radius
num_mts = tubulin_budget / cell_radius
optimization_temperature = 1e-3

growth_rate = 0.5 # um / s
catastrophe_rate = 0.025 # 1 / s
average_mt_length = growth_rate / catastrophe_rate # um

proportion_MTs_reaching_surface = 0.4

learning_rate = np.round(1 /  ((tubulin_budget / proportion_MTs_reaching_surface) * np.square(catastrophe_rate) / growth_rate ), 5)
# learning_rate = 0.01

spindle_length = 20 # um

print(f'learning rate: {learning_rate} s')

trimesh = np.load(f'{home}multi-aster/trimesh_cache/sphere_{mesh_density}_subdivs_1_radius.npy') * cell_radius
sloan_100 = np.load(f'{home}multi-aster/sloane_cache/sloane_{num_motors}.npy') * cell_radius

print(trimesh.shape)

# -- metaphase plate (disk) --
# radius relative to the cell, plus the drag coefficients from disk-forces.tex section 1 at a
# reference viscosity mu -- see Spindle.__init__ for the formulas (zeta_parallel = 32/3 mu R,
# zeta_perp = 16 mu R, zeta_omega = 32/3 mu R^3)
disk_radius = cell_radius / 4 # um
disk_viscosity = 1.0 # pN s um^-2
disk_zeta_parallel = 1e25 #(32 / 3) * disk_viscosity * disk_radius
disk_zeta_perp = 16 * disk_viscosity * disk_radius
disk_zeta_omega = (32 / 3) * disk_viscosity * disk_radius**3
num_disk_push_sites = 20000
num_disk_pull_sites = 23 # num sites on each face 23 * 2 = 46 kinetochores in total

# calculate_cost() wants the disk centred at the origin with its normal e1 horizontal, pointing
# along +x -- start it there, and put both MTOCs along that same x-axis (spindle poles on
# opposite sides of the plate, near the origin) so the optimizer spreads them out to
# spindle_length from a sensible, disk-centred starting configuration.
#
# disk_center is the starting R_D: displace it to watch the plate recentre from an off-axis
# start. It must satisfy |disk_center| + disk_radius <= cell_radius or Spindle rejects it.
disk_center = np.array([1.0, 1.0, 1.0])
disk_normal = np.array([2.0, 0.0, 0.0])
disk_tangent = np.array([0.0, 1.0, 0.0])
mtoc_positions = np.array([[10, 0, 0], [-10, 0, 0]])

force = 'both'

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
            disk_radius=disk_radius,
            disk_center=disk_center,
            disk_normal=disk_normal,
            disk_tangent=disk_tangent,
            num_disk_push_sites=num_disk_push_sites,
            num_disk_pull_sites=num_disk_pull_sites,
            disk_zeta_parallel=disk_zeta_parallel,
            disk_zeta_perp=disk_zeta_perp,
            disk_zeta_omega=disk_zeta_omega,
            disk_shadowing_enabled=True,
            fix_mtoc_positions=True, # MTOCs held fixed at mtoc_positions; the disk is left free to move
            data_dir=f'{ceph}multi_aster_disk/inital_experiments/',
            # dir_prefix=f'pull_disk_centring_{disk_radius}um_disk_{len(mtoc_positions)}_aster_{average_mt_length}_lbar_{force}_{tubulin_budget}_tubulin__{1e-4}_dt_{optimization_temperature}_temperature_{learning_rate}_learning_rate',
            dir_prefix=f'fixed_mtocs_{disk_radius}um_disk_{len(mtoc_positions)}_aster_{average_mt_length}_lbar_{force}_{tubulin_budget}_tubulin_{1e-4}_dt_{optimization_temperature}_temperature_{learning_rate}_learning_rate',
            save_trajectory=True,
            save=True
            )

spindle.optimize(num_attempts, max_lab_time=max_time)

spindle.plot_cost()
spindle.plot_aster_distance_from_centre()
if len(mtoc_positions) == 2:
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
print(f'disk sites occupied: {num_disk_occupied} / {2 * (num_disk_push_sites + num_disk_pull_sites)}')

# animate_mtoc_trajectory already overlays the disk (its pose, characteristic vectors e1/e2/e3,
# and site occupancy) alongside the MTOCs -- see multi_aster_spindle.py
ani_path = os.path.join(spindle.plot_folder_path, f'occupancy_ani.mp4')
spindle.animate_mtoc_trajectory(save_path=ani_path, interval=50, stride=100, show_occupancy=force, occupancy_mtoc_id=1, start_time=0, end_time=max_time)
