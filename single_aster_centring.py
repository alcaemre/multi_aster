#
# Emre Alca
# University of Pennsylvania
# Created on Mon Nov 23 2026
#
# Single-aster (centrosome) centring. One MTOC starts off-centre in a spherical cell with no
# metaphase plate, and the cost rewards it for sitting at the cell centre (the 'centring'
# term, which targets the origin when the disk is disabled). The optimizer chooses which
# pushing / pulling MTs to attach, and the aster moves under the forces they exert.
#
# With one MTOC, calculate_cost_terms drops the two-aster terms ('spatial' and
# 'spindle_disk_alignment'), and with the disk disabled it drops the four disk terms, so the
# cost is centring + material + motors. Plots are regenerated every plot_every accepted
# states in the experiment's plots/ folder, as in optimize_and_plot.py.
#

import numpy as np

import spindle as mas

import matplotlib.pyplot as plt
import os
import traceback


num_attempts = 50000000
max_time = 5000 # s of simulated time

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# -- cell and lattice --
mesh_density = 7
cell_radius = 20 # um
tubulin_budget = 10000 # um
target_num_motors = 200 # desired number of engaged cortical motors (occupied boundary pull sites)
optimization_temperature = 6e-3

growth_rate = 0.5 # um / s
catastrophe_rate = 0.05 # 1 / s
average_mt_length = growth_rate / catastrophe_rate # um

learning_rate = growth_rate / (tubulin_budget * np.square(catastrophe_rate))

print(f'learning rate: {learning_rate} s')

trimesh = np.load(f'{home}multi-aster/trimesh_cache/sphere_{mesh_density}_subdivs_1_radius.npy') * cell_radius

# -- the aster --
# a single MTOC, started off-centre so the run shows it being brought to the middle.
# Must lie inside the cell (|position| < cell_radius).
mtoc_positions = np.array([[8.0, 3.0, -4.0]])

force = 'both'

# -- cost function prefactors --
# only centring, material and motors are evaluated for a single aster with the disk disabled;
# the other keys are listed so the saved spindle.pkl records every prefactor, and are unused.
cost_coefficients = {
    'centring': 10,                       # aster position vs cell centre
    'material': 1,                        # tubulin use vs tubulin_budget
    'motors': 1,                          # engaged cortical motors vs target_num_motors
    'spatial': 0,                         # (two asters only) spindle length
    'spindle_disk_alignment': 0,          # (two asters only) spindle axis vs disk normal
    'disk_position': 0,                   # (disk only)
    'disk_orientation': 0,                # (disk only)
    'disk_normal_angular_velocity': 0,    # (disk only)
    'disk_com_velocity': 0,               # (disk only)
}

spindle = mas.Spindle(initial_mtoc_positions=mtoc_positions,
            lattice=trimesh,
            boundary_radius=cell_radius,
            tubulin_budget=tubulin_budget,
            target_num_motors=target_num_motors,
            growth_rate=growth_rate,
            spindle_length=10, # unused with one aster
            stall_force=10.0,
            rigidity=30,
            pull_force=10.0,
            average_mt_length=average_mt_length,
            optimization_temperature=optimization_temperature,
            evolution_time=learning_rate,
            euler_timestep_size=learning_rate / 100,
            disk_enabled=False,
            disk_shadowing_enabled=False,
            cost_coefficients=cost_coefficients,
            seed=1,
            fix_mtoc_positions=False,
            data_dir=f'{ceph}single-aster-centring',
            dir_prefix=f'cat_{catastrophe_rate}_centring_{cost_coefficients["centring"]:g}_material_{cost_coefficients["material"]:g}_motors_{cost_coefficients["motors"]:g}_{average_mt_length}_lbar_{force}_{tubulin_budget}_tubulin_{target_num_motors}_motors_{optimization_temperature}_temperature_{learning_rate}_learning_rate',
            save_trajectory=True,
            save=True
            )

# -- plotting --
plot_every = 10000 # accepted states between plot refreshes
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
