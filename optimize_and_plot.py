#
# Emre Alca
# University of Pennsylvania
# Created on Sat Oct 03 2026
#
# Optimizes a spindle exactly like disk_centring.py, but regenerates the full set of
# analysis plots from animate_occupancy.py every plot_every accepted states, overwriting
# the previous set in the experiment's plots/ folder. No animation is rendered.
#

import numpy as np

import spindle as mas

import matplotlib.pyplot as plt
import argparse
import os
import traceback


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

num_attempts = 50000000
max_time = 5000

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# making push/pull lattices with 10 um radii
# kept smaller than two-aster-centring.py's mesh_density=8 (655k points) -- with that many
# boundary sites, the disk's own sites (a few hundred at most) would almost never win the
# weighted draw in sample_spindle_update purely on numbers, regardless of relative distance.
mesh_density = 7
cell_radius = 20 # um
tubulin_budget = 10000 #num_mts*cell_radius
target_num_motors = 200 # desired number of engaged cortical motors (occupied boundary pull sites)
num_mts = tubulin_budget / cell_radius
optimization_temperature = 6e-3

growth_rate = 0.5 # um / s
catastrophe_rate = 0.05 # 1 / s
average_mt_length = growth_rate / catastrophe_rate # um

proportion_MTs_reaching_surface = 0.36

learning_rate = np.round(1 /  ((tubulin_budget / proportion_MTs_reaching_surface) * np.square(catastrophe_rate) / growth_rate ), 5)

learning_rate = growth_rate / (tubulin_budget * np.square(catastrophe_rate))
# learning_rate = 0.01

spindle_length = 10 # um

print(f'learning rate: {learning_rate} s')

trimesh = np.load(f'{home}multi-aster/trimesh_cache/sphere_{mesh_density}_subdivs_1_radius.npy') * cell_radius
# sloan_100 = np.load(f'{home}multi-aster/sloane_cache/sloane_{num_motors}.npy') * cell_radius

# -- metaphase plate (disk) --
# radius relative to the cell, plus the drag coefficients from disk-forces.tex section 1 at a
# reference viscosity mu -- see Spindle.__init__ for the formulas (zeta_parallel = 32/3 mu R,
# zeta_perp = 16 mu R, zeta_omega = 32/3 mu R^3)
disk_radius = cell_radius / 4 # um
disk_viscosity = 1.0 # pN s um^-2
disk_zeta_parallel = (32 / 3) * disk_viscosity * disk_radius
disk_zeta_perp = 16 * disk_viscosity * disk_radius
disk_zeta_omega = (32 / 3) * disk_viscosity * disk_radius**3
num_disk_push_sites = 20000
num_disk_pull_sites = 20000 # num sites on each face 23 * 2 = 46 kinetochores in total

# calculate_cost() wants the disk centred at the origin with its normal e1 horizontal, pointing
# along +x -- start it there, and put both MTOCs along that same x-axis (spindle poles on
# opposite sides of the plate, near the origin) so the optimizer spreads them out to
# spindle_length from a sensible, disk-centred starting configuration.
#
# disk_center is the starting R_D: displace it to watch the plate recentre from an off-axis
# start. It must satisfy |disk_center| + disk_radius <= cell_radius or Spindle rejects it.
disk_center = np.array([2.0, -5.0, 6.0])
disk_normal = mas.normalize_vecs(np.array([1.0, 1.0, 1.0]))[0]
disk_tangent = np.array([0.0, 1.0, 0.0])
mtoc_positions = np.array([[5, 2, 6], [-3, -10, 3]])

force = 'both'

# -- cost function prefactors --
# every term of Spindle.calculate_cost, spelled out so each run's choice is visible here and
# saved in spindle.pkl (any key left out falls back to mas.DEFAULT_COST_COEFFICIENTS)
cost_coefficients = {
    'spatial': 50,                         # spindle length vs spindle_length
    'material': 1,                      # tubulin use vs tubulin_budget
    'motors': 1,                        # engaged cortical motors vs target_num_motors
    'centring': 0,                        # MTOC centroid vs disk centre
    'disk_position':100,                   # disk centre of mass vs origin
    'disk_orientation': 100,                # disk normal e1 vs lab +x
    'spindle_disk_alignment': 0,          # spindle axis vs disk normal e1
    'disk_normal_angular_velocity': 0, # tumble rate of the disk normal
    'disk_com_velocity': 0,               # speed of the disk centre of mass
}

spindle = mas.Spindle(initial_mtoc_positions=mtoc_positions,
            lattice=trimesh,
            boundary_radius=cell_radius,
            tubulin_budget=tubulin_budget,
            target_num_motors=target_num_motors,
            growth_rate=growth_rate,
            spindle_length=spindle_length,
            stall_force=10.0,
            rigidity=30,
            pull_force=10.0,
            average_mt_length=average_mt_length,
            optimization_temperature=optimization_temperature,
            evolution_time=learning_rate,
            euler_timestep_size=learning_rate / 100,
            disk_radius=disk_radius,
            disk_center=disk_center,
            disk_normal=disk_normal,
            disk_tangent=disk_tangent,
            num_disk_push_sites=num_disk_push_sites,
            num_disk_pull_sites=num_disk_pull_sites,
            disk_zeta_parallel=disk_zeta_parallel,
            disk_zeta_perp=disk_zeta_perp,
            disk_zeta_omega=disk_zeta_omega,
            disk_enabled=True,
            disk_shadowing_enabled=True,
            cost_coefficients=cost_coefficients,
            seed=1,
            fix_mtoc_positions=False, # MTOCs held fixed at mtoc_positions; the disk is left free to move
            data_dir=f'{ceph}parameter-check', #multi-aster-parallel',#multi_aster_disk/one_lattice', #one_lattice',
            # dir_prefix=f'pull_disk_centring_{disk_radius}um_disk_{len(mtoc_positions)}_aster_{average_mt_length}_lbar_{force}_{tubulin_budget}_tubulin__{1e-4}_dt_{optimization_temperature}_temperature_{learning_rate}_learning_rate',
            dir_prefix=f'cat_{catastrophe_rate}_prefactors_{"_".join(f"{v:g}" for v in cost_coefficients.values())}_{len(mtoc_positions)}_aster_{average_mt_length}_lbar_{force}_{tubulin_budget}_tubulin_{target_num_motors}_motors_{optimization_temperature}_temperature_{learning_rate}_learning_rate',
            save_trajectory=True,
            save=True
            )

# -- plotting --
plot_every = 10000 # accepted spindle states between plot refreshes
occupancy_window = 100 # s, window over which occupancy and lifetimes are averaged


def safe_make_plots(spindle):
    # a plotting failure should cost one set of plots, not a multi-hour run
    try:
        mas.make_plots(spindle, force=force, window=occupancy_window)
    except Exception:
        traceback.print_exc()
        plt.close('all')


# one proposal-evaluating worker per core SLURM gives the job (see Spindle.optimize / ProposalPool)
num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

spindle.optimize(num_attempts, max_lab_time=max_time, callback=safe_make_plots, callback_every=plot_every, num_workers=num_workers)

# final set of plots for wherever the run stopped
safe_make_plots(spindle)
