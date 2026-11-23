#
# Emre Alca
# University of Pennsylvania
# Created on Wed Jun 03 2026
# Last Modified: 2026/10/06 11:41:30
#

import numpy as np
np.set_printoptions(formatter={'float': '{:.3f}'.format})
# import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import time
from datetime import datetime
import tqdm

import pickle
import os
import glob
import itertools
import contextlib
import gc
import multiprocessing as mp
import traceback
import psutil
from scipy.spatial import cKDTree

from disk_tesselation import sunflower_disk_polar

from rich.console import Console
from rich.live import Live
from rich.table import Table
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

console = Console()

# prefactors of the terms in Spindle.calculate_cost. Spindle(cost_coefficients=...) overrides
# any subset of these; the merged dict is stored on the spindle (and so in spindle.pkl).
DEFAULT_COST_COEFFICIENTS = {
    'spatial': 0,                         # spindle length vs spindle_length
    'material': 0.1,                      # tubulin use vs tubulin_budget
    'motors': 0,                          # engaged cortical motors vs target_num_motors
    'centring': 0,                        # MTOC centroid vs disk centre
    'disk_position': 1,                   # disk centre of mass vs origin
    'disk_orientation': 1,                # disk normal e1 vs lab +x
    'spindle_disk_alignment': 0,          # spindle axis vs disk normal e1
    'disk_normal_angular_velocity': 0.01, # tumble rate of the disk normal
    'disk_com_velocity': 0.01,            # speed of the disk centre of mass
}


def safe_pickle_dump(obj, target_path):
    """
    Write to a temp file first, then atomically rename to target.
    If the process is killed mid-write, the original file is untouched
    and the temp file is simply left incomplete/orphaned.
    """
    tmp_path = os.path.abspath(target_path) + ".tmp"
    try:
        with open(tmp_path, "wb") as tmp_f:
            pickle.dump(obj, tmp_f)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    # Atomic rename — this is the key step
    os.replace(tmp_path, target_path)  # use os.replace(), not os.rename()


def safe_np_save(arr, target_path):
    """
    np.save counterpart of safe_pickle_dump: write to a temp file, then atomically
    rename, so a reader running alongside the simulation (plot_progress.py) never
    loads a half-written .npy.
    """
    tmp_path = os.path.abspath(target_path) + ".tmp"
    try:
        # a file object, not a path: np.save appends .npy to any path that lacks it
        with open(tmp_path, "wb") as tmp_f:
            np.save(tmp_f, arr)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    os.replace(tmp_path, target_path)


def claim_experiment_dir(data_dir, dir_prefix):
    """
    Creates and returns a new experiment directory under data_dir, never reusing one
    that already exists.

    Two runs with the same dir_prefix (e.g. the same cost prefactors) used to share a
    directory and silently overwrite each other's traces. Now the first run gets
    data_dir/dir_prefix; a later one gets dir_prefix_<SLURM job id> (a timestamp off
    the cluster), then _2, _3, ... if even that is taken. os.mkdir is atomic, so two
    jobs starting at the same moment can't both claim the same name.
    """
    os.makedirs(data_dir, exist_ok=True)

    tag = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidates = itertools.chain([dir_prefix, f'{dir_prefix}_{tag}'],
                                 (f'{dir_prefix}_{tag}_{n}' for n in itertools.count(2)))
    for name in candidates:
        path = os.path.join(data_dir, name)
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        if name != dir_prefix:
            print(f'{os.path.join(data_dir, dir_prefix)} already exists -- saving this run to {path} instead')
        return path


def trace_batch_name(kind, start, end):
    """Filename for a saved batch of accepted states.

    Args:
        kind (str): 'push' or 'pull'.
        start (int): number of accepted states before this batch.
        end (int): number of accepted states after this batch.

    Returns:
        str: e.g. 'push_states_0_1000.npy'.
    """
    return f'{kind}_states_{start}_{end}.npy'


def load_trajectory(experiment_dir):
    """
    Reassemble an experiment's trajectory from disk.

    Reads the legacy single-file trajectory.pkl (if present) and then every
    trajectory/trajectory_{start}_{end}.pkl chunk written by optimize(), in order
    of start, merging them into one dict keyed by time.

    Returns:
        dict: time -> timepoint data, in the order the timepoints were accepted.
    """
    trajectory = {}

    # experiments saved before the trajectory was chunked
    legacy_path = os.path.join(experiment_dir, 'trajectory.pkl')
    if os.path.exists(legacy_path):
        with open(legacy_path, 'rb') as f:
            trajectory.update(pickle.load(f))

    chunk_files = sorted(
        glob.glob(os.path.join(experiment_dir, 'trajectory', 'trajectory_*.pkl')),
        key=lambda f: int(os.path.basename(f).split('_')[1])
    )
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            trajectory.update(pickle.load(f))

    return trajectory


def normalize_vecs(vecs):
    """
    Normalizes an array of vectors

    Args:
        vecs (numpy.array): array of vectors to normalize

    Returns:
        numpy.array: normalized vecs
    """

    if vecs.shape == (3,):
        norm = np.linalg.norm(vecs)
        saved_norm = norm.copy()
        if norm == 0: # avoid division by zero
            norm = 1
        return vecs / norm, saved_norm
    else:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)

        saved_norms = norms.copy().flatten()

        norms[norms == 0] = 1  # Avoid division by zero
        return vecs / norms, saved_norms


def disk_local_tessellation(n, radius):
    """
    Body-frame (a, b) Cartesian coordinates for one face of a disk's tessellation,
    from the (r, theta) sunflower tessellation in disk_tesselation.py.

    Args:
        n (int): number of lattice sites.
        radius (float): disk radius.

    Returns:
        np.ndarray: shape (n, 2) array of (a, b) coordinates.
    """
    points = sunflower_disk_polar(n, radius)
    if not points:
        return np.zeros((0, 2))
    r, theta = np.array(points).T
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def build_push_to_pull_map(push_points, pull_points, motor_radius):
    """
    Maps each push-lattice point to its nearest pull-lattice point, keeping only
    pairs within motor_radius. Used to "promote" a sampled push site to a pull
    attachment when a motor is close enough -- shared by the boundary lattice
    and the disk's own push/pull tessellations.

    Args:
        push_points (np.ndarray): shape (N, d) push lattice/tessellation points.
        pull_points (np.ndarray): shape (M, d) pull lattice/tessellation points.
        motor_radius (float): maximum distance to keep a push -> pull pairing.

    Returns:
        dict[int, int]: push index -> nearest pull index, only for pairs within motor_radius.
    """
    tree = cKDTree(pull_points)
    distances, nearest_pull_idx = tree.query(push_points, k=1)
    return {
        push_idx: pull_idx
        for push_idx, (pull_idx, dist) in enumerate(zip(nearest_pull_idx, distances))
        if dist <= motor_radius
    }


def rotate_frame(vectors, omega, dt):
    """
    Rotates a set of vectors by angle |omega|*dt about axis omega/|omega|, via
    Rodrigues' rotation formula. Used to advance a rigid body's co-rotating
    frame under Euler integration without accumulating orthonormality drift.

    Args:
        vectors (np.ndarray): shape (K, 3) vectors to rotate.
        omega (np.ndarray): shape (3,) angular velocity vector.
        dt (float): timestep.

    Returns:
        np.ndarray: shape (K, 3) rotated vectors.
    """
    angle = np.linalg.norm(omega) * dt
    if angle == 0:
        return vectors.copy()

    axis = omega / np.linalg.norm(omega)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    return (vectors * cos_a
            + np.cross(axis, vectors) * sin_a
            + axis * np.dot(vectors, axis)[:, np.newaxis] * (1 - cos_a))


def retrieve_experiement(experiment_dir, save_trajectory=False, save=False, max_states=None):

    num_attempts_path = os.path.join(experiment_dir, 'num_attempts.npy')
    num_attempts = np.load(num_attempts_path)

    num_accepted_states_path = os.path.join(experiment_dir, 'num_accepted_states.npy')
    num_accepted_states = np.load(num_accepted_states_path)

    # -- reinitialize spindle from saved spindle dict and trajectory --
    spindle_path = os.path.join(experiment_dir, 'spindle.pkl')
    with open(spindle_path, 'rb') as f:
        spindle_dict = pickle.load(f)

    trajectory = load_trajectory(experiment_dir)

    last_time = list(trajectory.keys())[-1]
    last_mtoc_positions = np.array(list(trajectory[last_time]['mtoc_pos'].values()))

    push_state = np.zeros(spindle_dict['push_lattice'].shape[0])
    pull_state = np.zeros(spindle_dict['pull_lattice'].shape[0])

    num_disk_push_sites = spindle_dict.get('num_disk_push_sites', 50)
    num_disk_pull_sites = spindle_dict.get('num_disk_pull_sites', 20)
    disk_push_state_front = np.zeros(num_disk_push_sites)
    disk_push_state_back = np.zeros(num_disk_push_sites)
    disk_pull_state_front = np.zeros(num_disk_pull_sites)
    disk_pull_state_back = np.zeros(num_disk_pull_sites)

    spindle_trace_dir = os.path.join(experiment_dir, 'spindle_trace')
    trace_files = sorted(
        glob.glob(os.path.join(spindle_trace_dir, 'spindle_trace_*.npy')),
        key=lambda f: int(os.path.basename(f).split('_')[2])
    )

    for trace_file in trace_files:
        for push, lattice_site, site_value in np.load(trace_file, allow_pickle=True):
            if push:
                push_state[int(lattice_site)] = site_value
            else:
                pull_state[int(lattice_site)] = site_value

    # metaphase plate (disk) attachments: a separate, parallel trace log (see optimize())
    disk_trace_files = sorted(
        glob.glob(os.path.join(spindle_trace_dir, 'disk_trace_*.npy')),
        key=lambda f: int(os.path.basename(f).split('_')[2])
    )

    for trace_file in disk_trace_files:
        for push, front, lattice_site, site_value in np.load(trace_file, allow_pickle=True):
            if push:
                state = disk_push_state_front if front else disk_push_state_back
            else:
                state = disk_pull_state_front if front else disk_pull_state_back
            state[int(lattice_site)] = site_value

    # reconstruct trajectory

    spindle_from_dict = Spindle(
        initial_mtoc_positions=last_mtoc_positions,
        lattice=spindle_dict['push_lattice'],
        # pull_lattice=spindle_dict['pull_lattice'],
        initial_time=last_time, # s
        fix_mtoc_positions=spindle_dict.get('fix_mtoc_positions', False),

        # -- optimization paramters --
        tubulin_budget=spindle_dict['tubulin_budget'], # µm
        target_num_motors=spindle_dict.get('target_num_motors', 100), # default for experiments saved before this existed
        num_attempts=num_attempts, # only set if restarting an experiment
        num_accepted_states=num_accepted_states, # only set if restarting an experiment
    
        # -- hyperparameters --
        optimization_temperature=spindle_dict['optimization_temperature'], # temperature in Metropolis-Hastings style simulated annealing
        euler_timestep_size=spindle_dict['euler_timestep_size'], # s
        evolution_time=spindle_dict['evolution_time'], # s
    
        # -- biophysical constants --
        
        rigidity=spindle_dict['rigidity'], # pN µm^2 
        sliding_friction_coefficient=spindle_dict['sliding_friction_coefficient'], # pN s µm^{−1} coefficient of friction of MT sliding along cytoskeleton
        growth_rate=spindle_dict['growth_rate'], # µm s^{−1}
        pull_force=spindle_dict['pull_force'], # pN 
        stall_force=spindle_dict['stall_force'], # pN
        cytoplasmic_drag_factor=spindle_dict['cytoplasmic_drag_factor'], # pN s µm^{−1} drag factor of aster
        boundary_radius=spindle_dict['boundary_radius'], # µm
        motor_radius=spindle_dict['motor_radius'], # µm
        average_mt_length=spindle_dict['average_mt_length'],
        spindle_length = spindle_dict['spindle_length'],

        # -- metaphase plate (disk) config -- (spindle_dict.get(...) falls back to the
        # constructor defaults for experiments saved before the disk feature existed)
        # default True: experiments saved before this flag existed all had a live plate
        disk_enabled=spindle_dict.get('disk_enabled', True),
        disk_radius=spindle_dict.get('disk_radius', 2.0),
        # the pose the run itself last held, so a restart resumes the disk where it was even
        # when the trajectory carries no disk snapshot (see the fallback below)
        disk_center=spindle_dict.get('disk_center'),
        disk_normal=spindle_dict.get('disk_e1'),
        disk_tangent=spindle_dict.get('disk_e2'),
        num_disk_push_sites=num_disk_push_sites,
        num_disk_pull_sites=num_disk_pull_sites,
        disk_zeta_parallel=spindle_dict.get('disk_zeta_parallel', 21.33),
        disk_zeta_perp=spindle_dict.get('disk_zeta_perp', 32.0),
        disk_zeta_omega=spindle_dict.get('disk_zeta_omega', 85.33),
        disk_shadowing_enabled=spindle_dict.get('disk_shadowing_enabled', False),

        # -- cost function -- (None for experiments saved before cost_coefficients existed,
        # which then fall back to DEFAULT_COST_COEFFICIENTS)
        cost_coefficients=spindle_dict.get('cost_coefficients'),

        # -- saving info --
        seed=spindle_dict['seed'],  # random seed
        dir_path=experiment_dir, # directory of existing experiment
        data_dir = spindle_dict['path_to_data'], # directory within which each experiment's data is stored
        save_trajectory=save_trajectory, # True if you want the system to save the whole trajectory. Used to make animations and plots
        save=save,
        trajectory=trajectory,
        )

    # the constructor seeded rng with the run's original seed, which would make a restart
    # replay the random stream the run began with. Mixing in how far the run had got gives
    # each restart point its own stream, still reproducible from the same seed.
    if spindle_dict['seed'] is not None:
        spindle_from_dict.rng = np.random.default_rng([int(spindle_dict['seed']), int(num_attempts)])

    spindle_from_dict.push_state = push_state
    spindle_from_dict.pull_state = pull_state

    spindle_from_dict.disk_push_state_front = disk_push_state_front
    spindle_from_dict.disk_push_state_back = disk_push_state_back
    spindle_from_dict.disk_pull_state_front = disk_pull_state_front
    spindle_from_dict.disk_pull_state_back = disk_pull_state_back

    last_disk_data = trajectory[last_time]
    if not spindle_from_dict.disk_enabled:
        pass  # no plate in this experiment; nothing to restore and nothing to warn about
    elif 'disk_center' in last_disk_data:
        spindle_from_dict.disk_center = last_disk_data['disk_center']
        spindle_from_dict.disk_e1 = last_disk_data['disk_e1']
        spindle_from_dict.disk_e2 = last_disk_data['disk_e2']
        spindle_from_dict.disk_e3 = last_disk_data['disk_e3']
    else:
        print("No disk pose was recorded in this trajectory (predates disk tracking) -- "
              "restarting the disk at the pose saved in spindle.pkl "
              f"(centre {spindle_from_dict.disk_center}, e1 {spindle_from_dict.disk_e1}).")

    return spindle_from_dict
    

def make_plots(spindle, force='both', window=100):
    """
    Regenerates the standard set of plots for a run so far, overwriting the previous set
    in spindle.plot_folder_path. Shared by optimize_and_plot.py, restart_experiment.py and
    plot_progress.py so they all draw the same thing.

    Args:
        spindle (Spindle): the run, with its trajectory loaded (in memory or via
            retrieve_experiement).
        force (str): 'push', 'pull' or 'both' -- which occupancy and lifetime maps to draw.
        window (float): s, occupancy and lifetimes are averaged over the last window
            seconds of the trajectory.
    """
    end_time = max(spindle.trajectory.keys())
    start_time = max(end_time - window, 0)

    print(f'\nplotting at {spindle.num_accepted_states} accepted states / {spindle.num_attempts} attempts '
          f'(acceptance rate {spindle.num_accepted_states / max(spindle.num_attempts, 1):.4f}), t = {end_time:.3f} s')

    spindle.plot_cost(end_time=end_time)
    spindle.plot_cost_terms(end_time=end_time)
    spindle.plot_mt_length(end_time=end_time)
    spindle.plot_aster_distance_from_centre(end_time=end_time)
    if len(spindle.mtoc_positions) == 2:
        spindle.plot_aster_separation(end_time=end_time)
        spindle.plot_spindle_orientation(end_time=end_time)
    if spindle.disk_enabled:
        spindle.plot_disk_distance_from_centre(end_time=end_time)
        spindle.plot_disk_orientation(end_time=end_time)
    spindle.plot_num_mts_per_mtoc(force=force, end_time=end_time)
    if spindle.disk_enabled:
        spindle.plot_num_disk_mts_per_mtoc(force=force, end_time=end_time)
    spindle.plot_pull_to_push_ratio()

    # occupancy and lifetime, by angle and over the surface
    spindle.plot_occupancy_vs_angle(1, start_time, end_time, show_occupancy=force)
    spindle.plot_lifetime_vs_angle(1, start_time, end_time, show_occupancy=force)
    for occupancy in (['pull', 'push'] if force == 'both' else [force]):
        spindle.plot_surface_occupancy(1, start_time, end_time, show_occupancy=occupancy)
        spindle.plot_lifetime_surface(1, start_time, end_time, show_occupancy=occupancy)

    # latest snapshot, under a fixed name so it is overwritten (the default name embeds t)
    frame_path = os.path.join(spindle.plot_folder_path, 'mtoc_trajectory_frame_latest.png')
    spindle.plot_mtoc_trajectory_frame(end_time, 5, save_path=frame_path, show_occupancy=True)

    print(f'disk centre: {spindle.disk_center}  (target: origin)')
    print(f'disk e1 (normal): {spindle.disk_e1}  (target: +x)')
    num_disk_occupied = (np.count_nonzero(spindle.disk_push_state_front) + np.count_nonzero(spindle.disk_push_state_back)
                          + np.count_nonzero(spindle.disk_pull_state_front) + np.count_nonzero(spindle.disk_pull_state_back))
    print(f'disk sites occupied: {num_disk_occupied} / {2 * (spindle.num_disk_push_sites + spindle.num_disk_pull_sites)}')
    print(f'plots written to {spindle.plot_folder_path}')


class ProposalPool:
    """
    Worker processes that evaluate Metropolis-Hastings proposals in parallel for
    Spindle.optimize(num_workers=K).

    A rejected proposal is fully undone, so every proposal made between two accepted
    states is an independent trial from the same state. Each round, worker i (i = 0..K-1)
    makes and evaluates one proposal from the current state on its own copy of the
    spindle, and the round's winner is the accepted proposal with the lowest i -- the one
    the serial loop would have reached first had it drawn the same K proposals in that
    order. A parallel run therefore has the same statistics as a serial one whatever K is;
    only the random streams differ. Picking the lowest-cost acceptance instead, or the first
    worker to finish, would bias the dynamics (towards downhill moves, or towards moves that
    are quick to evaluate) and make results depend on K.

    Workers are forked from the coordinating process, so each starts as an exact copy of the
    spindle. Each round's winner sends back the state it produced, and every copy applies
    that rather than re-integrating it, so the copies stay identical.
    """

    def __init__(self, spindle, num_workers):
        ctx = mp.get_context('fork')
        # an independent random stream per worker, derived from the spindle's own stream (not
        # its seed) so that a restarted run's workers don't replay the original run's streams
        seeds = np.random.SeedSequence(int(spindle.rng.integers(2**63))).spawn(num_workers)

        self.connections = []
        self.processes = []
        self.pending_change = None # last round's winner, sent out with the next dispatch

        # the coordinating process can hold a large in-memory trajectory. Freezing the garbage
        # collector across the fork keeps each worker's collector from walking those objects,
        # which would copy every page they sit on into the worker.
        gc.freeze()
        try:
            for seed in seeds:
                connection, worker_connection = ctx.Pipe()
                process = ctx.Process(target=_proposal_worker, args=(spindle, worker_connection, seed), daemon=True)
                process.start()
                worker_connection.close()
                self.connections.append(connection)
                self.processes.append(process)
        finally:
            gc.unfreeze()

    def dispatch(self, old_cost):
        """Starts a round: every worker applies the last winner, then evaluates one proposal."""
        for connection in self.connections:
            connection.send((self.pending_change, old_cost))
        self.pending_change = None

    def collect(self):
        """
        Waits for the round started by dispatch.

        Returns:
            dict or None: the winning change (see Spindle._apply_accepted_change), or None if
                no worker's proposal was accepted.
            int: how many proposals count as attempted -- up to and including the winner,
                or all of them if there wasn't one.
        """
        replies = [connection.recv() for connection in self.connections]
        for reply in replies:
            if isinstance(reply, str):
                raise RuntimeError(f'a proposal worker failed:\n{reply}')

        for i, change in enumerate(replies):
            if change is not None:
                self.pending_change = change
                return change, i + 1
        return None, len(replies)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        for connection in self.connections:
            try:
                connection.send(None)
            except (BrokenPipeError, OSError):
                pass # worker already gone
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        return False


def _proposal_worker(spindle, connection, seed):
    """
    Event loop of one ProposalPool worker. Each round: apply the last round's winner,
    make one proposal, evaluate it, undo it, and report the resulting change if it was
    accepted (None otherwise). A traceback string is sent back if anything raises.
    """
    spindle.rng = np.random.default_rng(seed)
    # only the coordinating process writes anything. save_trajectory=False also stops
    # time_evolution from costing every Euler step for a trajectory nothing here keeps.
    spindle.save = False
    spindle.save_trajectory = False

    try:
        while True:
            message = connection.recv()
            if message is None:
                break
            change, old_cost = message

            if change is not None:
                spindle._apply_accepted_change(change)
                spindle.time = spindle.time + spindle.evolution_time

            proposal = spindle.sample_spindle_update()
            if proposal is None:
                connection.send(None)
                continue

            evaluation = spindle._evaluate_proposal(proposal, old_cost)
            spindle._revert_proposal(evaluation)

            if evaluation['accepted']:
                connection.send({key: evaluation[key] for key in ('proposal', 'cost', 'cost_terms', 'culled', 'mtoc_positions', 'disk_pose')})
            else:
                connection.send(None)
    except EOFError:
        pass # coordinating process exited
    except Exception:
        connection.send(traceback.format_exc())


class Spindle:

    def __init__(
            self, 
            initial_mtoc_positions, # numpy array of shape (k,3) for k MTOCs
            lattice,
            initial_time=0.0,
            fix_mtoc_positions=False, # if True, MTOCs never move during time_evolution -- the
                                       # disk (and site occupancy) still evolves normally

            # -- optimization paramters --
            tubulin_budget=100.0, # µm
            target_num_motors=100, # desired number of engaged cortical motors (occupied boundary pull sites)
            num_attempts=0, # only set if restarting an experiment
            num_accepted_states=0, # only set if restarting an experiment

            # -- hyperparameters --
            optimization_temperature=(1e-4), # temperature in Metropolis-Hastings style simulated annealing
            evolution_time=(1e-3), # s Derived from the nucleation rate of MTs being 500 s^-1, 
                                   # and we have nucleation and catastrophe rates equal at steady state, 
                                   # then, every 1/500 s, an MT is added and an MT is removed.
            euler_timestep_size=(1e-4), # s so that there are at least 10 euler steps between spindle updates

            # -- biophysical constants --
            
            rigidity=30.0, # pN µm^2 
            sliding_friction_coefficient=1.0, # pN s µm^{−1} coefficient of friction of MT sliding along cytoskeleton
            growth_rate=0.5, # µm s^{−1}
            pull_force=10.0, # pN 
            stall_force=10.0, # pN
            cytoplasmic_drag_factor=150.0, # pN s µm^{−1} drag factor of aster
            boundary_radius=10.0, # µm
            motor_radius=1.5, # µm
            average_mt_length=10, # µm
            spindle_length=13, # µm

            # -- metaphase plate (disk) --
            # False removes the plate from the simulation entirely: no disk sites, no disk
            # forces, no shadowing, and no disk terms in calculate_cost. Every disk-specific
            # plot and the animation's disk overlay skip themselves too. Use this to run the
            # boundary-only experiments this code did before the plate existed.
            disk_enabled=True,
            disk_radius=2.0, # µm
            disk_center=None, # (3,) initial R_D, defaults to the origin
            disk_normal=None, # (3,) initial e1, defaults to +z
            disk_tangent=None, # (3,) initial in-plane reference vector used to build e2, defaults to +x
            num_disk_push_sites=50,
            num_disk_pull_sites=20,
            # defaults below correspond to the disk-forces.tex formulas at µ=1.0 pN s µm^-2, radius=2.0 µm,
            # but are independent constructor params (not derived from radius/viscosity at runtime)
            disk_zeta_parallel=21.33, # pN s µm^-1, resists in-plane (radial/tangential) motion
            disk_zeta_perp=32.0, # pN s µm^-1, resists broadside (normal) motion
            disk_zeta_omega=85.33, # pN s µm, resists spin and tumble
            # opt-in: sample_spindle_update only rejects disk-shadowed boundary sites when this
            # is True, so existing experiments that don't care about the disk are unaffected by
            # the fact that a (small, default) disk always exists on the Spindle
            disk_shadowing_enabled=False,

            # -- cost function --
            # dict of prefactors for the calculate_cost terms; any keys left out take their
            # value from DEFAULT_COST_COEFFICIENTS
            cost_coefficients=None,

            # -- saving info --
            trajectory={},
            seed=None,  # random seed
            dir_path=None, # directory of existing experiment
            dir_prefix='', # name of this experiment
            data_dir = None, # directory within which each experiment's data is stored
            save_trajectory=False, # True if you want the system to save the whole trajectory. Used to make animations and plots
            save=False,
            ):
        """
        Initializes a spindle 

        Args:
            initial_mtoc_positions (np.ndarray): Initial MTOC positions, shape (k, 3) for k MTOCs.
            push_lattice           (np.ndarray): Boundary push sites, shape (N, 3).
            pull_lattice           (np.ndarray): Boundary pull sites (motor positions), shape (M, 3).
            initial_time                (float): Simulation start time in seconds.

            tubulin_budget              (float): Total MT length budget in µm.
            target_num_motors             (int): Desired number of occupied boundary pull sites (motors).
            num_attempts                  (int): Attempted state changes; set when restarting an experiment.
            num_accepted_states           (int): Accepted states; set when restarting an experiment.

            cost_tolerance              (float): Costs below this threshold are treated as zero.
            euler_timestep_size         (float): Euler integration timestep in seconds.
            spindle_update_delay_time   (float): Simulated time between spindle configuration updates in seconds.

            rigidity                    (float): MT bending rigidity in pN µm².
            sliding_friction_coefficient (float): Friction coefficient for MT sliding along the cytoskeleton in pN s µm⁻¹.
            growth_rate                 (float): MT growth rate in µm s⁻¹.
            pull_force                  (float): Motor pull force in pN.
            stall_force                 (float): Motor stall force in pN.
            cytoplasmic_drag_factor     (float): Drag factor for aster movement through cytoplasm in pN s µm⁻¹.
            boundary_radius             (float): Cell boundary radius in µm.

            cost_coefficients     (dict or None): Prefactors of the calculate_cost terms, keyed as in
                                                  DEFAULT_COST_COEFFICIENTS; missing keys use the defaults.

            seed                 (int or None): Random seed for reproducibility.
            dir_path              (str or None): Path to an existing experiment directory; set when restarting.
            dir_prefix                   (str): Prefix for the auto-generated experiment directory name.
            data_dir              (str or None): Parent directory where experiment directories are created.
            save_trajectory              (bool): If True, stores the full trajectory in memory for animations.
        """
        
        # -- setting physical parameters --
        self.pull_force = pull_force
        self.stall_force = stall_force
        self.rigidity = rigidity 
        self.sliding_friction_coefficient = sliding_friction_coefficient
        self.growth_rate = growth_rate
        self.cytoplasmic_drag_factor = cytoplasmic_drag_factor
        self.boundary_radius = boundary_radius
        self.motor_radius = motor_radius  # persisted so retrieve_experiement can reconstruct the spindle
        self.average_mt_length = average_mt_length
        self.spindle_length = spindle_length

        # -- setting optimization parameters --
        self.tubulin_budget = tubulin_budget
        self.target_num_motors = target_num_motors
        self.num_attempts = num_attempts
        self.num_accepted_states = num_accepted_states

        # -- setting hyperparameters --
        self.euler_timestep_size = euler_timestep_size
        self.evolution_time = evolution_time 
        # self.cost_tolerance = cost_tolerance 
        self.optimization_temperature = optimization_temperature

        # -- setting cost function prefactors --
        cost_coefficients = dict(cost_coefficients or {})
        unknown_terms = set(cost_coefficients) - set(DEFAULT_COST_COEFFICIENTS)
        if unknown_terms:
            raise ValueError(f"unknown cost_coefficients keys {sorted(unknown_terms)}; "
                             f"valid keys are {sorted(DEFAULT_COST_COEFFICIENTS)}")
        self.cost_coefficients = {**DEFAULT_COST_COEFFICIENTS, **cost_coefficients}

        # -- setting parameters --
        self.time = initial_time
        
        # initialize lattices
        self.push_lattice = lattice
        self.pull_lattice = lattice

        # initialize spindle states:
        # this uses the convention where push_lattice[i] = 0 means the site is empty 
        # and push_lattice[j]=k means that there is an MT from aster k impinging on site j
        # similar for push_lattice
        self.num_push_sites = lattice.shape[0]
        self.num_pull_sites = lattice.shape[0]

        self.push_state = np.zeros(self.num_push_sites)
        self.pull_state = np.zeros(self.num_pull_sites)

        # only set while time_evolution runs, see _held_sites
        self._held_sites_cache = None

        self.push_boundary_unit_normals = normalize_vecs(lattice)[0] # only matters for calculating pushing force magnitude

        # initialize position of MTOCs

        mtoc_positions = {}

        num_asters = len(initial_mtoc_positions)
        for aster in range(num_asters):
            mtoc_positions[aster+1] = initial_mtoc_positions[aster]
        
        self.mtoc_positions = mtoc_positions
        self.fix_mtoc_positions = fix_mtoc_positions

        # set up map from pushing lattice to pulling lattice
        self.push_to_pull = build_push_to_pull_map(lattice, lattice, motor_radius)

        # -- metaphase plate (disk) --
        self.disk_enabled = disk_enabled
        self.disk_radius = disk_radius
        self.disk_zeta_parallel = disk_zeta_parallel
        self.disk_zeta_perp = disk_zeta_perp
        self.disk_zeta_omega = disk_zeta_omega
        # shadowing is geometric -- boundary_sites_accessible tests rays against disk_radius,
        # not against the site counts -- so a disabled plate would still occlude the boundary
        # unless this is forced off here
        self.disk_shadowing_enabled = disk_shadowing_enabled and disk_enabled

        # with no sites, every disk state array is empty: the sampler has nothing to draw, the
        # force sums are empty, and the tubulin/MT counts pick up nothing. The pose vectors are
        # still built below so the rest of the class has valid (if unused) e1/e2/e3.
        if not disk_enabled:
            num_disk_push_sites = 0
            num_disk_pull_sites = 0

        if disk_center is None:
            disk_center = np.zeros(3)
        if disk_normal is None:
            disk_normal = np.array([0.0, 0.0, 1.0])
        if disk_tangent is None:
            disk_tangent = np.array([1.0, 0.0, 0.0])

        self.disk_center = np.asarray(disk_center, dtype=float)
        if self.disk_center.shape != (3,):
            raise ValueError(f"disk_center must have shape (3,), got {self.disk_center.shape}")

        # A disk that starts outside the cell trips time_evolution's boundary check on the very
        # first Euler step, so every proposal is rejected and optimize() breaks out after one
        # attempt with no diagnostic. Same test as in time_evolution -- fail here instead.
        # Skipped when the plate is disabled: there is then no plate to collide with, and
        # time_evolution does not run the check either.
        if disk_enabled and np.linalg.norm(self.disk_center) + disk_radius > boundary_radius:
            raise ValueError(
                f"disk starts outside the cell: |disk_center| = {np.linalg.norm(self.disk_center):.3f} µm "
                f"+ disk_radius = {disk_radius} µm exceeds boundary_radius = {boundary_radius} µm"
            )

        self.disk_e1 = normalize_vecs(np.asarray(disk_normal, dtype=float))[0]

        # Gram-Schmidt disk_tangent against e1 to get an orthonormal in-plane reference vector
        disk_tangent = np.asarray(disk_tangent, dtype=float)
        disk_tangent = disk_tangent - np.dot(disk_tangent, self.disk_e1) * self.disk_e1
        # a tangent parallel to the normal leaves nothing in-plane; normalize_vecs would hand back
        # a zero e2 (and so a zero e3), collapsing the body frame without complaint
        if np.linalg.norm(disk_tangent) == 0:
            if disk_enabled:
                raise ValueError("disk_tangent is parallel to disk_normal, so it defines no in-plane direction")
            # no plate to orient, so don't make the caller supply a tangent they don't care
            # about -- pick any perpendicular and keep the frame orthonormal
            seed_axis = np.array([0.0, 1.0, 0.0]) if abs(self.disk_e1[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
            disk_tangent = seed_axis - np.dot(seed_axis, self.disk_e1) * self.disk_e1
        self.disk_e2 = normalize_vecs(disk_tangent)[0]
        self.disk_e3 = np.cross(self.disk_e1, self.disk_e2)

        # body-frame (a, b) coordinates of each face's tessellation, shared by front and back;
        # only the occupancy state differs per face
        self.num_disk_push_sites = num_disk_push_sites
        self.num_disk_pull_sites = num_disk_pull_sites
        self.disk_push_local = disk_local_tessellation(num_disk_push_sites, disk_radius)
        self.disk_pull_local = disk_local_tessellation(num_disk_pull_sites, disk_radius)

        # push -> pull promotion map, same on both faces since they share the same local geometry
        self.disk_push_to_pull_front = build_push_to_pull_map(self.disk_push_local, self.disk_pull_local, motor_radius)
        self.disk_push_to_pull_back = self.disk_push_to_pull_front

        # occupancy state: 0 = empty, mtoc_id = an MT from that mtoc is impinging on that site.
        # front = +e1 outward normal, back = -e1 outward normal.
        self.disk_push_state_front = np.zeros(num_disk_push_sites)
        self.disk_push_state_back = np.zeros(num_disk_push_sites)
        self.disk_pull_state_front = np.zeros(num_disk_pull_sites)
        self.disk_pull_state_back = np.zeros(num_disk_pull_sites)

        # -- setting up data saving --

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.path_to_data = data_dir # path to the parent directory within which we make the director for each experiment

        self.save = save

        if dir_path is None and save: # meaning a directory for the experiment has not already been made
            if dir_prefix == '':
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                dir_prefix = f'{num_asters}_asters_{tubulin_budget}_um_tubulin_{boundary_radius}_um_radius_{timestamp}'
            # always a fresh directory -- never another run's, even with the same dir_prefix
            self.dir_path = claim_experiment_dir(data_dir, dir_prefix)

            # make folder for plots
            self.plot_folder_path = os.path.join(self.dir_path, 'plots')
            os.makedirs(self.plot_folder_path, exist_ok=True)

            # make spindle_trace folder
            self.spindle_trace_path = os.path.join(self.dir_path, 'spindle_trace')
            os.makedirs(self.spindle_trace_path, exist_ok=True) # raises error if data directory does not exist

            # make trajectory folder (optimize() writes one chunk per save batch)
            self.trajectory_path = os.path.join(self.dir_path, 'trajectory')
            os.makedirs(self.trajectory_path, exist_ok=True)

            # save spindle params as dict
            spindle_dict = dict(self.__dict__)

            spindle_path = os.path.join(self.dir_path, 'spindle.pkl')
            safe_pickle_dump(spindle_dict, spindle_path)
            print(f'spindle saved to {spindle_path}')

            # save num_accepted_states
            safe_np_save(self.num_accepted_states, os.path.join(self.dir_path, 'num_accepted_states.npy'))

            # save num_total_attempts
            safe_np_save(self.num_attempts, os.path.join(self.dir_path, 'num_attempts.npy'))

        else:
            self.dir_path = dir_path
            self.plot_folder_path = os.path.join(self.dir_path, 'plots')
            self.spindle_trace_path = os.path.join(self.dir_path, 'spindle_trace')
            self.trajectory_path = os.path.join(self.dir_path, 'trajectory')
            if save:
                # experiments saved before the trajectory was chunked have no trajectory folder
                os.makedirs(self.trajectory_path, exist_ok=True)

        self.save_trajectory = save_trajectory
        # if save_trajectory:
        # the trajectory is persisted in chunks by optimize(); see load_trajectory()
        self.trajectory = trajectory


    def add_microtubules(self, mtoc_id, mt_indices_to_add, push):
        """
        Places a MTs connecting either the pushing or pulling lattice with indices mt_indices_to_add to the MTOC with mtoc_id

        Args:
            mtoc_id (int): id of MTOC MTs are nucleating from
            mt_indices_to_add (np.array(ints)): indices of lattice points MTs are impinging on to
            push (bool): true if MTs are being added to pushing lattice, False if MTS are being added to pulling lattice

        Raises:
            ValueError: cannot add a microtubule to a site already containing a microtubule
            ValueError: the provided mtoc_id does not refer to an mtoc in this spindle
        """

        if push:
            state = self.push_state
        else:
            state = self.pull_state

        if len(np.where(state[mt_indices_to_add] != 0)[0]) != 0:
            raise ValueError("cannot add a microtubule to a site already containing a microtubule")
        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        state[mt_indices_to_add] = mtoc_id
        

    def remove_microtubules(self, mt_indices_to_remove, push):
        """_summary_

        Args:
            mt_indices_to_rmove (np.array(ints)): indices of lattice points MTs are being removed from
            push (bool): true if MTs are being added to pushing lattice, False if MTS are being added to pulling lattice

        Raises:
            ValueError: cannot remove a microtubule from an empty site
        """

        if push:
            state = self.push_state
        else:
            state = self.pull_state

        if len(np.where(state[mt_indices_to_remove] == 0)[0]) != 0:
            raise ValueError("cannot remove a microtubule from an empty site")

        state[mt_indices_to_remove] = 0

    def add_microtubules_to_disk(self, mtoc_id, site_indices, push, front):
        """
        Manually attaches MTs to metaphase-plate (disk) lattice sites. Intended for
        exercising the disk's equations of motion directly; not yet wired into
        sample_spindle_update / optimize.

        Args:
            mtoc_id (int): id of MTOC MTs are nucleating from
            site_indices (np.array(ints)): indices of disk lattice points MTs are impinging on
            push (bool): True for the disk's pushing lattice, False for its pulling lattice
            front (bool): True for the +e1 face, False for the -e1 face

        Raises:
            ValueError: cannot add a microtubule to a site already containing a microtubule
            ValueError: the provided mtoc_id does not refer to an mtoc in this spindle
        """

        if push:
            state = self.disk_push_state_front if front else self.disk_push_state_back
        else:
            state = self.disk_pull_state_front if front else self.disk_pull_state_back

        if len(np.where(state[site_indices] != 0)[0]) != 0:
            raise ValueError("cannot add a microtubule to a site already containing a microtubule")
        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        state[site_indices] = mtoc_id

    def remove_microtubules_from_disk(self, site_indices, push, front):
        """
        Manually removes MTs from metaphase-plate (disk) lattice sites.
        See add_microtubules_to_disk.

        Args:
            site_indices (np.array(ints)): indices of disk lattice points to clear
            push (bool): True for the disk's pushing lattice, False for its pulling lattice
            front (bool): True for the +e1 face, False for the -e1 face

        Raises:
            ValueError: cannot remove a microtubule from an empty site
        """

        if push:
            state = self.disk_push_state_front if front else self.disk_push_state_back
        else:
            state = self.disk_pull_state_front if front else self.disk_pull_state_back

        if len(np.where(state[site_indices] == 0)[0]) != 0:
            raise ValueError("cannot remove a microtubule from an empty site")

        state[site_indices] = 0

    def _resolve_state_array(self, is_disk, push, front):
        """
        Returns the actual state array a sample_spindle_update() descriptor refers to,
        one of push_state/pull_state/disk_push_state_{front,back}/disk_pull_state_{front,back}.
        Used by optimize() to mutate/revert the one site a proposed change targets.
        """
        if not is_disk:
            return self.push_state if push else self.pull_state
        if push:
            return self.disk_push_state_front if front else self.disk_push_state_back
        return self.disk_pull_state_front if front else self.disk_pull_state_back

    def _disk_lab_frame(self, local_xy, face_sign):
        """
        Maps body-frame (a, b) disk-tessellation coordinates to their current
        lab-frame geometry (disk-forces.tex section 1).

        Args:
            local_xy (np.ndarray): shape (K, 2) body-frame (a, b) coordinates.
            face_sign (float): +1.0 for the front face (outward normal +e1), -1.0 for back (-e1).

        Returns:
            lab_positions (np.ndarray): shape (K, 3) current lab-frame positions.
            r (np.ndarray): shape (K,) in-plane radial distance from the disk centre.
            n_hat, s_hat, t_hat (np.ndarray): shape (K, 3) unit normal, radial, and
                tangential directions at each site.
        """
        a = local_xy[:, 0]
        b = local_xy[:, 1]

        in_plane = a[:, np.newaxis] * self.disk_e2 + b[:, np.newaxis] * self.disk_e3
        lab_positions = self.disk_center + in_plane

        r = np.hypot(a, b)
        s_hat = in_plane / r[:, np.newaxis]
        n_hat = np.tile(face_sign * self.disk_e1, (len(a), 1))
        t_hat = np.cross(n_hat, s_hat)

        return lab_positions, r, n_hat, s_hat, t_hat

    def disk_ray_intersection(self, ray_origin, directions):
        """
        Ray/disk intersection test (disk-forces.tex section 2): for MT(s) leaving
        ray_origin along unit vector(s) `directions`, finds where -- if at all --
        the straight-line path crosses the metaphase plate.

        Args:
            ray_origin (np.ndarray): shape (3,), the MT's starting point (R_M).
            directions (np.ndarray): shape (K, 3) unit vectors (m_hat), or shape
                (3,) for a single direction.

        Returns:
            hits (np.ndarray or bool): shape (K,) (or scalar for a single
                direction) -- True where the ray crosses the disk's plane within
                its radius, at a positive parameter t.
            t_star (np.ndarray or float): shape (K,) (or scalar) -- the crossing
                parameter (MT length) along each ray; only meaningful where
                `hits` is True.
        """
        single = (directions.ndim == 1)
        dirs = directions[np.newaxis, :] if single else directions

        R = ray_origin - self.disk_center
        m_dot_n = dirs @ self.disk_e1

        with np.errstate(divide='ignore', invalid='ignore'):
            t_star = -np.dot(R, self.disk_e1) / m_dot_n
        # m_dot_n == 0 means the MT travels parallel to the disk's plane and never
        # crosses it; use 0 rather than +/-inf so hit_points below stays finite
        # (t_star > 0 already excludes this case from `hits`).
        t_star = np.where(m_dot_n == 0, 0.0, t_star)

        hit_points = ray_origin + t_star[:, np.newaxis] * dirs
        within_radius = np.linalg.norm(hit_points - self.disk_center, axis=1) <= self.disk_radius
        hits = (t_star > 0) & within_radius

        if single:
            return bool(hits[0]), float(t_star[0])
        return hits, t_star

    def is_disk_face_reachable(self, mtoc_id, front):
        """
        True if MTOC `mtoc_id` sits on the side of the disk's (infinitesimally
        thin) plane matching `front` -- i.e. it can grow a straight MT directly
        onto that face of the metaphase plate without passing through the disk.

        Args:
            mtoc_id (int): id of the MTOC.
            front (bool): True to check the +e1 face, False for the -e1 face.

        Raises:
            ValueError: the provided mtoc_id does not refer to an mtoc in this spindle

        Returns:
            bool: True if the MTOC is on the correct side to reach that face.
        """
        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        side = np.dot(self.mtoc_positions[mtoc_id] - self.disk_center, self.disk_e1)
        return side > 0 if front else side < 0

    def boundary_sites_accessible(self, mtoc_id, site_positions):
        """
        For each candidate site (typically boundary lattice points), checks
        whether a straight MT from MTOC `mtoc_id` can reach it without first
        being blocked by the metaphase plate's shadow (disk-forces.tex section 2).

        Args:
            mtoc_id (int): id of the MTOC.
            site_positions (np.ndarray): shape (K, 3) candidate site positions.

        Raises:
            ValueError: the provided mtoc_id does not refer to an mtoc in this spindle

        Returns:
            np.ndarray: shape (K,) boolean array, True where the site is accessible.
        """
        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        mtoc_position = self.mtoc_positions[mtoc_id]
        dirs, distances = normalize_vecs(site_positions - mtoc_position)

        hits, t_star = self.disk_ray_intersection(mtoc_position, dirs)
        shadowed = hits & (t_star < distances)

        return ~shadowed

    def cull_shadowed_microtubules(self):
        """
        Detaches every attached MT whose straight line from its own MTOC is no longer
        clear, and reports what was cleared.

        Reachability is otherwise only ever checked at nucleation, when
        sample_spindle_update picks a site that is lit at that moment. The metaphase
        plate then keeps moving under its own EOM, so a site that was lit when its MT
        grew there can sit behind the plate a few steps later -- and without this the MT
        stays attached, still pushing or pulling through the plate. Two ways that
        happens, one per lattice family:
          - a boundary site the plate has since moved in front of;
          - a disk site whose MTOC has ended up on the far side of the plate, so that
            face can no longer be reached in a straight line at all.

        Does nothing unless disk_shadowing_enabled, so experiments that don't model the
        plate's occlusion keep their previous behaviour (see __init__).

        Returns:
            list: (is_disk, push, front, lattice_site, old_site_value) per MT detached
            -- enough for optimize() to unwind the cull if the step is rejected, and to
            log it to the spindle traces if it is accepted.
        """
        if not (self.disk_enabled and self.disk_shadowing_enabled):
            return []

        cleared = []

        # -- boundary lattices: has the plate moved into the line of sight? --
        for push, state, lattice in ((True, self.push_state, self.push_lattice),
                                      (False, self.pull_state, self.pull_lattice)):
            occupied = np.flatnonzero(state)
            if occupied.size == 0:
                continue
            holders = state[occupied]
            for mtoc_id in np.unique(holders):
                held = occupied[holders == mtoc_id]
                blocked = held[~self.boundary_sites_accessible(int(mtoc_id), lattice[held])]
                for site in blocked:
                    cleared.append((False, push, None, int(site), state[site]))
                    state[site] = 0

        # -- disk faces: an MTOC can only hold a face it is still on the near side of --
        for push, front, state in ((True, True, self.disk_push_state_front),
                                    (True, False, self.disk_push_state_back),
                                    (False, True, self.disk_pull_state_front),
                                    (False, False, self.disk_pull_state_back)):
            occupied = np.flatnonzero(state)
            if occupied.size == 0:
                continue
            holders = state[occupied]
            for mtoc_id in np.unique(holders):
                if self.is_disk_face_reachable(int(mtoc_id), front=front):
                    continue
                for site in occupied[holders == mtoc_id]:
                    cleared.append((True, push, front, int(site), state[site]))
                    state[site] = 0

        return cleared

    def calculate_pulling_forces(self, mtoc_id):
        """
        Calculates the pulling force experienced by the MTOC.
        For the moment, we assume the pulling force to be constant

        Args:
            mtoc_id (int): id of MTOC we are calculating the force on

        Raises:
            ValueError: the provided mtoc_id does not refer to an mtoc in this spindle

        Returns:
            float: pulling force in pN experienced by the MTOC
        """

        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        # -- finding relevant m-hats --
        # find lattice point with MTs connected to mtoc_id
        connected_points = self.pull_lattice[self._held_sites(self.pull_state, mtoc_id)]

        relevant_mt_vecs = connected_points - self.mtoc_positions[mtoc_id]
        dirs, norms = normalize_vecs(relevant_mt_vecs)

        return self.pull_force * np.sum(dirs, axis=0)


    def calculate_pushing_forces(self, mtoc_id):
        """
        calculates the pushing force experienced by the MTOC.

        Args:
            mtoc_id (int): id of MTOC we are calculating the force on

        Raises:
            ValueError: the provided mtoc_id does not refer to an mtoc in this spindle

        Returns:
            float: pushing force in pN experienced by the MTOC
        """

        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        # -- finding relevant m-hats --
        # find lattice point with MTs connected to mtoc_id
        held = self._held_sites(self.push_state, mtoc_id)
        connected_points = self.push_lattice[held]
        relevant_mt_vecs = connected_points - self.mtoc_positions[mtoc_id]
        dirs, norms = normalize_vecs(relevant_mt_vecs)

        pushing_boundary_normals = self.push_boundary_unit_normals[held]
        pushing_force_magnitudes = self._pushing_force_magnitudes(dirs, norms, pushing_boundary_normals)

        # total pushing force is the component-wise sum of the pushing vectors
        pushing_vectors = pushing_force_magnitudes[:, np.newaxis] * dirs

        # -- summing over pushing force vectors to find total pushing force --
        # pushing_mt_dirs point outwards from the mtoc, we want pushing forces to point inwards towards the mtoc
        total_pushing_force = -np.sum(pushing_vectors, axis=0)

        return total_pushing_force

    def _pushing_force_magnitudes(self, dirs, norms, wall_normals):
        """
        Shared force-magnitude model for an MT pushing against a wall (boundary or
        disk): a stall-force ratchet, capped by the Euler buckling force.

        Args:
            dirs (np.ndarray): shape (K, 3) unit vectors from MTOC to each pushing site.
            norms (np.ndarray): shape (K,) MT lengths.
            wall_normals (np.ndarray): shape (K, 3) outward unit normal of the wall at each site.

        Returns:
            np.ndarray: shape (K,) pushing force magnitudes.
        """
        # -- calculating buckling forces --
        buckling_forces = (np.pi**2) * self.rigidity / (norms**2)

        # -- calculating unbuckled pushing forces --
        # calculating the effective force coefficients (mt_dir . wall_norm)
        effective_force_coefficients = np.sum(dirs * wall_normals, axis=1)
        # calculating the denominator of the pushing force magnitude
        pushing_force_denominators = (self.stall_force / (self.growth_rate * self.sliding_friction_coefficient)) * (1 - effective_force_coefficients) + 1
        # putting the pieces together
        pushing_force_magnitudes = self.stall_force / pushing_force_denominators

        # -- pushing forces are bounded above by the buckling force --
        pushing_force_magnitudes[pushing_force_magnitudes > buckling_forces] = buckling_forces[pushing_force_magnitudes > buckling_forces]

        return pushing_force_magnitudes

    def calculate_disk_pulling_forces(self, mtoc_id):
        """
        Reaction force on an MTOC from pulling MTs attached to the metaphase plate.

        Args:
            mtoc_id (int): id of MTOC we are calculating the force on

        Raises:
            ValueError: the provided mtoc_id does not refer to an mtoc in this spindle

        Returns:
            np.ndarray: shape (3,) pulling force in pN experienced by the MTOC
        """
        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        total_force = np.zeros(3)
        if not self.disk_enabled:
            return total_force

        for state, face_sign in [(self.disk_pull_state_front, 1.0), (self.disk_pull_state_back, -1.0)]:
            occupied = self._held_sites(state, mtoc_id)
            if occupied.size == 0:
                continue

            lab_positions, _, _, _, _ = self._disk_lab_frame(self.disk_pull_local[occupied], face_sign)
            dirs, _ = normalize_vecs(lab_positions - self.mtoc_positions[mtoc_id])
            total_force += self.pull_force * np.sum(dirs, axis=0)

        return total_force

    def calculate_disk_pushing_forces(self, mtoc_id):
        """
        Reaction force on an MTOC from pushing MTs attached to the metaphase plate.

        Args:
            mtoc_id (int): id of MTOC we are calculating the force on

        Raises:
            ValueError: the provided mtoc_id does not refer to an mtoc in this spindle

        Returns:
            np.ndarray: shape (3,) pushing force in pN experienced by the MTOC
        """
        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        total_force = np.zeros(3)
        if not self.disk_enabled:
            return total_force

        for state, face_sign in [(self.disk_push_state_front, 1.0), (self.disk_push_state_back, -1.0)]:
            occupied = self._held_sites(state, mtoc_id)
            if occupied.size == 0:
                continue

            lab_positions, _, _, _, _ = self._disk_lab_frame(self.disk_push_local[occupied], face_sign)
            dirs, norms = normalize_vecs(lab_positions - self.mtoc_positions[mtoc_id])
            wall_normals = np.tile(face_sign * self.disk_e1, (len(dirs), 1))
            magnitudes = self._pushing_force_magnitudes(dirs, norms, wall_normals)
            total_force += -np.sum(magnitudes[:, np.newaxis] * dirs, axis=0)

        return total_force

    def calc_mtoc_velocity(self, mtoc_id):
        """Calculates the velocity of an mtoc based on the mtoc position and the set of pushing and pulling mts connected to it.
        This is the implementation of the mtoc's equation of motion.

        Returns:
            numpy.array: velocity vector in the form of numpy.array([x,y,z])
        """
        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")

        return (self.calculate_pulling_forces(mtoc_id) + self.calculate_pushing_forces(mtoc_id)
                + self.calculate_disk_pulling_forces(mtoc_id) + self.calculate_disk_pushing_forces(mtoc_id)
                ) / self.cytoplasmic_drag_factor

    def calculate_disk_velocity_and_omega(self):
        """
        Equations of motion for the metaphase plate (disk-forces.tex section 1),
        aggregated over every MT currently attached to any disk site, from any MTOC.

        Returns:
            U (np.ndarray): shape (3,) translational velocity of the disk centre.
            omega (np.ndarray): shape (3,) angular velocity of the disk frame.
        """
        U = np.zeros(3)
        omega = np.zeros(3)

        if not self.disk_enabled:
            return U, omega

        groups = [
            (self.disk_push_state_front, self.disk_push_local, 1.0, True),
            (self.disk_push_state_back, self.disk_push_local, -1.0, True),
            (self.disk_pull_state_front, self.disk_pull_local, 1.0, False),
            (self.disk_pull_state_back, self.disk_pull_local, -1.0, False),
        ]

        for state, local, face_sign, is_push in groups:
            occupied = self._held_sites(state)
            if occupied.size == 0:
                continue

            lab_pos, r, n_hat, s_hat, t_hat = self._disk_lab_frame(local[occupied], face_sign)
            mtoc_pos = np.array([self.mtoc_positions[mtoc_id] for mtoc_id in state[occupied]])
            dirs, norms = normalize_vecs(lab_pos - mtoc_pos)

            if is_push:
                # reaction to the MTOC-side "-sum(magnitude * dirs)"
                wall_normals = np.tile(face_sign * self.disk_e1, (len(dirs), 1))
                magnitudes = self._pushing_force_magnitudes(dirs, norms, wall_normals)
                force_on_disk = magnitudes[:, np.newaxis] * dirs
            else:
                # reaction to the MTOC-side "+pull_force * dirs"
                force_on_disk = -self.pull_force * dirs

            f_n = np.sum(force_on_disk * n_hat, axis=1)
            f_r = np.sum(force_on_disk * s_hat, axis=1)
            f_t = np.sum(force_on_disk * t_hat, axis=1)

            U += np.sum(f_n[:, np.newaxis] * n_hat, axis=0) / self.disk_zeta_perp
            U += np.sum(f_r[:, np.newaxis] * s_hat + f_t[:, np.newaxis] * t_hat, axis=0) / self.disk_zeta_parallel
            omega += np.sum(r[:, np.newaxis] * (f_t[:, np.newaxis] * n_hat - f_n[:, np.newaxis] * t_hat), axis=0) / self.disk_zeta_omega

        return U, omega


    def time_evolution(self, evolution_time=None):
        """evolve time in the system by some amount.
        Progresses time for all MTOCs.

        Args:
            time_to_evolve_by (float): amount of time to evolve by in seconds

        Returns:
            dict, bool: MTOC positions after time evolution, whether the MTOCs or the disk
            left the cell along the way
        """
        if evolution_time is None:
            evolution_time = self.evolution_time

        boundary_violated = False

        # a fixed number of Euler steps, e.g. exactly 10 for euler_timestep_size =
        # evolution_time / 10. Looping until the summed steps reached evolution_time instead
        # let floating-point round-off add an 11th step to a large fraction of proposals.
        num_steps = max(1, int(round(evolution_time / self.euler_timestep_size)))

        mtoc_positions = self.mtoc_positions.copy()
        starting_mtoc_positions = self.mtoc_positions

        # no site is attached or detached during the integration -- only the MTOCs and the
        # disk move -- so which sites each MTOC holds is looked up once here rather than
        # every Euler step (see _held_sites)
        self._held_sites_cache = {}
        try:
            for _ in range(num_steps):
                if boundary_violated:
                    break

                # explicit Euler: every velocity in this step comes from the state at its start.
                # The force functions read self.mtoc_positions, so point it at the positions
                # integrated so far -- otherwise every sub-step would reuse the forces at the
                # positions the integration started from.
                self.mtoc_positions = mtoc_positions

                if not self.fix_mtoc_positions:
                    mtoc_velocities = {mtoc_id: self.calc_mtoc_velocity(mtoc_id) for mtoc_id in mtoc_positions}
                if self.disk_enabled:
                    U_disk, omega_disk = self.calculate_disk_velocity_and_omega()

                # evolve each MTOC by one timestep, unless they've been fixed in place (the disk
                # still evolves below regardless).
                if not self.fix_mtoc_positions:
                    for mtoc_id, dr_dt in mtoc_velocities.items():
                        # calculate the new position of the MTOC
                        mtoc_positions[mtoc_id] = mtoc_positions[mtoc_id] + (dr_dt * self.euler_timestep_size)

                        # check that the new mtoc position is not outside of the radius
                        normalized_new_mtoc_pos, new_mtoc_pos_norm = normalize_vecs(mtoc_positions[mtoc_id])
                        if new_mtoc_pos_norm > self.boundary_radius:
                            boundary_violated = True

                # evolve the metaphase plate by one timestep (mutated in place -- not yet
                # reverted on optimize()'s Metropolis-Hastings rejection, see plan notes)
                if self.disk_enabled:
                    self.disk_center = self.disk_center + U_disk * self.euler_timestep_size

                    frame = rotate_frame(np.stack([self.disk_e1, self.disk_e2, self.disk_e3]),
                                          omega_disk, self.euler_timestep_size)
                    self.disk_e1, self.disk_e2, self.disk_e3 = frame

                    if np.linalg.norm(self.disk_center) + self.disk_radius > self.boundary_radius:
                        boundary_violated = True
        finally:
            # the caller decides whether to keep the new positions (they are returned)
            self.mtoc_positions = starting_mtoc_positions
            self._held_sites_cache = None

        return mtoc_positions, boundary_violated

    def _held_sites(self, state, mtoc_id=None):
        """
        Indices, in increasing order, of the sites in one occupancy state array held by
        mtoc_id -- or by any MTOC, when mtoc_id is None. Equivalent to the boolean mask
        state == mtoc_id (state != 0) for indexing, so results are unchanged, but inside
        time_evolution each answer is computed once and reused by every Euler step
        rather than rescanning the whole lattice each time.
        """
        cache = self._held_sites_cache
        key = (id(state), mtoc_id)
        if cache is not None and key in cache:
            return cache[key]

        held = np.flatnonzero(state) if mtoc_id is None else np.flatnonzero(state == mtoc_id)
        if cache is not None:
            cache[key] = held
        return held
    
    def calculate_tubulin_use_per_mtoc(self):
        """
        Total length of MTs emanating from each MTOC, to both the boundary and the
        metaphase plate (disk).

        Returns:
            dict[int, float]: mtoc_id -> total MT length (um)
        """
        mt_length_per_mtoc = {}
        # for each MTOC
        mtoc_ids = np.array(list(self.mtoc_positions.keys()))
        for id in mtoc_ids:
            # find the set of lattice points with an MT connecting that point with that MTOC
            push_points_connected_to_id = self.push_lattice[self.push_state==id]
            pull_points_connected_to_id = self.pull_lattice[self.pull_state==id]

            # subtract the MTOC position
            push_vecs = (push_points_connected_to_id - self.mtoc_positions[id])
            pull_vecs = (pull_points_connected_to_id - self.mtoc_positions[id])

            # find the norms
            push_norms = normalize_vecs(push_vecs)[1]
            pull_norms = normalize_vecs(pull_vecs)[1]

            # the sums of these norms are this MTOC's MT length
            mt_length = np.sum(push_norms) + np.sum(pull_norms)

            # metaphase plate (disk): MTs attached there draw from the same tubulin budget
            for state, local, face_sign in [
                (self.disk_push_state_front, self.disk_push_local, 1.0),
                (self.disk_push_state_back, self.disk_push_local, -1.0),
                (self.disk_pull_state_front, self.disk_pull_local, 1.0),
                (self.disk_pull_state_back, self.disk_pull_local, -1.0),
            ]:
                connected = state == id
                if not connected.any():
                    continue
                lab_pos, _, _, _, _ = self._disk_lab_frame(local[connected], face_sign)
                mt_length += np.sum(normalize_vecs(lab_pos - self.mtoc_positions[id])[1])

            mt_length_per_mtoc[int(id)] = float(mt_length)

        return mt_length_per_mtoc

    def calculate_tubulin_use(self):
        return sum(self.calculate_tubulin_use_per_mtoc().values())


    def calculate_num_mts(self):

        push_mts = np.where(self.push_state != 0)[0]
        pull_mts = np.where(self.pull_state != 0)[0]

        num_mts = len(push_mts) + len(pull_mts)
        num_mts += np.count_nonzero(self.disk_push_state_front)
        num_mts += np.count_nonzero(self.disk_push_state_back)
        num_mts += np.count_nonzero(self.disk_pull_state_front)
        num_mts += np.count_nonzero(self.disk_pull_state_back)

        return num_mts


    def calculate_cost(self):
        """
        Total cost: the sum over calculate_cost_terms() of each term times its prefactor in
        self.cost_coefficients (summed in the order calculate_cost_terms returns them).

        Returns:
            float: cost
        """
        cost = 0
        for name, value in self.calculate_cost_terms().items():
            cost += self.cost_coefficients[name] * value
        return cost

    def calculate_cost_terms(self):
        """Calculates each cost term, *without* its prefactor
        This cost function has four term types:
        1. a term penalizing the over or under use of tubulin;
        2. a term saying the spindle centre (MTOC centroid) wants to be at the disk centre
           (the origin when the disk is disabled);
        3. a collection of terms saying that each pair of MTOCs wants to be as far as possible from all other MTOCs.
        4. terms saying the metaphase plate (disk) wants its centre of mass at the origin and its
           normal e1 aligned with the lab +x axis (up to sign -- e1 and -e1 are the same plate).
        5. a term saying the spindle axis (MTOC 1 -> MTOC 2) wants to be parallel to the disk
           normal e1 (again up to sign).
        6. a term saying the disk normal e1 wants zero angular velocity (|omega x e1| = 0).
        7. a term saying the disk centre of mass wants zero velocity (|U| = 0).
        8. a term penalizing having more or fewer engaged cortical motors than target_num_motors.

        Returns:
            dict[str, float]: term name (a key of cost_coefficients) -> the term's dimensionless
                value before its prefactor. The four disk terms are absent when the plate is
                disabled. calculate_cost() is the prefactor-weighted sum of these.
        """
        terms = {}
        mtoc_ids = np.array(list(self.mtoc_positions.keys()))

        # --- explicitly set MTOC positions ---
        # single aster set to a particular position
        # desired_mtoc_positions = {1: np.array([5.0, 0, 0])}

        # cost += spatial_coefficient * normalize_vecs(self.mtoc_positions[1] - desired_mtoc_positions[1])[1] / self.boundary_radius
        # --- ---

        # --- spatial cost ---

        # -- desired spindle elongation --
        current_spindle_length = 0
        i, j = np.triu_indices(len(mtoc_ids), k=1)  # k=1 excludes diagonal (no (i,i) pairs)
        pairs = np.column_stack([mtoc_ids[i], mtoc_ids[j]])

        for pair in pairs:
            current_spindle_length += normalize_vecs((self.mtoc_positions[pair[0]] - self.mtoc_positions[pair[1]]))[1]
        current_spindle_length = current_spindle_length / len(pairs)

        # for two asters
        two_aster_spindle_length = normalize_vecs(self.mtoc_positions[1] - self.mtoc_positions[2])[1]
        current_spindle_length = two_aster_spindle_length

        terms['spatial'] = np.square(1 - (current_spindle_length / self.spindle_length))

        # -- metaphase plate (disk) placement/orientation --
        # desired: centre of mass at the origin, and e1 along the lab +x axis -- i.e. the disk
        # standing "vertically", broadside to the MTOC axis, which is the pose that casts the
        # largest shadow (boundary_sites_accessible).
        #
        # The penalty is the squared angle between e1 and that target axis, taken through abs():
        # a plate's orientation is its *plane*, so e1 and -e1 are the same pose and the cost has
        # to be even under e1 -> -e1. Scoring the (theta, phi) spherical angles of e1 against
        # (pi/2, 0) instead -- as this used to -- made edge-on a trap: it rated e1 = -x (a
        # perfectly broadside plate, normal merely flipped) at phi^2 = pi^2 ~ 9.87, four times
        # worse than any edge-on pose at ~2.47, so past 90 deg the whole hemisphere was a basin
        # whose floor sat *above* edge-on. phi is also discontinuous across the yz-plane and
        # undefined at e1 = +/-z. The form below matches the old one across the +x hemisphere,
        # but is smooth everywhere and increases monotonically as the shadow shrinks to nothing.
        #
        # Skipped entirely when the plate is disabled. These terms depend only on the pose, not
        # on how many MTs are attached, so a disabled (never-moving) plate would otherwise add a
        # constant to every state -- e.g. the default e1 = +z is edge-on and scores the full
        # disk_orientation coefficient, (2/pi * pi/2)^2 = 1 times the coefficient. That
        # constant cancels in the Metropolis-Hastings delta, but optimize() seeds old_cost with
        # len(push_state), so on a lattice smaller than the constant nothing is ever accepted
        # and the run exits after one attempt with no error.
        if self.disk_enabled:
            disk_normal_target = np.array([1.0, 0.0, 0.0])
            disk_com_distance = normalize_vecs(self.disk_center)[1]
            # angle from e1 to the nearest of +/-disk_normal_target, in [0, pi/2]
            disk_tilt = np.arccos(np.clip(np.abs(np.dot(self.disk_e1, disk_normal_target)), 0.0, 1.0))

            terms['disk_position'] = np.square(disk_com_distance / self.boundary_radius)
            terms['disk_orientation'] = np.square(2 * disk_tilt / np.pi)

            # -- disk normal angular velocity --
            # the normal wants to be stationary. e1 evolves as de1/dt = omega x e1, so only the
            # tumble component of omega (perpendicular to e1) moves the normal -- spin about e1
            # leaves it fixed and is not penalized. Nondimensionalized by the tumble rate a single
            # pulling MT at the rim, acting along e1, would produce: pull_force * disk_radius / disk_zeta_omega.
            disk_velocity, disk_omega = self.calculate_disk_velocity_and_omega()
            disk_normal_rate = np.linalg.norm(np.cross(disk_omega, self.disk_e1))
            disk_omega_scale = self.pull_force * self.disk_radius / self.disk_zeta_omega
            terms['disk_normal_angular_velocity'] = np.square(disk_normal_rate / disk_omega_scale)

            # -- disk centre of mass velocity --
            # the centre of mass wants to be stationary. All of U (in-plane and broadside) is
            # penalized. Nondimensionalized by the speed a single pulling MT, acting along e1,
            # would drag the disk broadside: pull_force / disk_zeta_perp.
            disk_com_speed = np.linalg.norm(disk_velocity)
            disk_velocity_scale = self.pull_force / self.disk_zeta_perp
            terms['disk_com_velocity'] = np.square(disk_com_speed / disk_velocity_scale)

            spindle_alignment_target = self.disk_e1
        else:
            spindle_alignment_target = np.array([1, 0, 0])

        # -- spindle axis / disk normal alignment --
        # the MTOC 1 -> MTOC 2 axis should pierce the plate broadside, i.e. be parallel to e1.
        # Same abs() treatment as disk_tilt: the axis direction (1->2 vs 2->1) and e1 vs -e1
        # are both arbitrary, so the penalty must be even under either flip. Coincident MTOCs
        # give a zero axis vector (normalize_vecs leaves it zero), which scores the maximal
        # (pi/2)^2 rather than NaN.
        spindle_axis = normalize_vecs(self.mtoc_positions[2] - self.mtoc_positions[1])[0]
        spindle_disk_misalignment = np.arccos(np.clip(np.abs(np.dot(spindle_axis, spindle_alignment_target)), 0.0, 1.0))

        terms['spindle_disk_alignment'] = np.square(spindle_disk_misalignment)

        # # -- spindle centring --
        # place the centre of the spindle at the centre of the disk (or at the origin when the
        # plate is disabled, since a disabled plate isn't physically there). Kept as the offset
        # of the summed positions, i.e. len(mtoc_ids) * (centroid - target), so the term's scale
        # matches the old origin-centred version.
        centring_target = self.disk_center if self.disk_enabled else np.zeros(3)
        sum_positions = np.zeros(3)
        for i in range(len(mtoc_ids)):
            sum_positions += self.mtoc_positions[mtoc_ids[i]] - centring_target

        terms['centring'] = np.square(normalize_vecs(sum_positions)[1] / (self.boundary_radius))

        # -- preferring zero net force (zero velocity) -- 
        # velocity = 0
        # for i in range(len(mtoc_ids)):
        #     mtoc_velocity = normalize_vecs(self.calc_mtoc_velocity(mtoc_ids[i]))[1]
        #     velocity += mtoc_velocity
        
        # cost +=  no_net_force_coefficient * np.square(velocity / self.boundary_radius)

        

        # # -- forced separation --
        # # each MTOC wants to be as close to the origin as possible
        # for id in mtoc_ids:
        #     mtoc_pos_norm = normalize_vecs(self.mtoc_positions[id])[1]
        # cost += np.square(mtoc_pos_norm / (self.boundary_radius))

        # # each pair of MTOCs want to maximize the distance between each other.
        # i, j = np.triu_indices(len(mtoc_ids), k=1)  # k=1 excludes diagonal (no (i,i) pairs)
        # pairs = np.column_stack([mtoc_ids[i], mtoc_ids[j]])

        # for pair in pairs:
        #     cost += separation_coefficient * np.square(2 - (normalize_vecs((self.mtoc_positions[pair[0]] - self.mtoc_positions[pair[1]]))[1] / (self.boundary_radius)))
        # -- --
        # --- --- 
        
        # --- material cost ---
        # material budget
        total_mt_length = self.calculate_tubulin_use()
        
        # add the material cost (1 - total_mt_length/tubulin_budget)^2
        terms['material'] = np.abs(1 - (total_mt_length / self.tubulin_budget))

        # --- motor cost ---
        # number of engaged cortical motors, i.e. occupied boundary pull sites (disk pull
        # sites are kinetochores, not motors, and are not counted)
        num_motors = np.count_nonzero(self.pull_state)

        # add the motor cost |1 - num_motors/target_num_motors|, same form as the material cost
        terms['motors'] = np.abs(1 - (num_motors / self.target_num_motors))

        # --- --- 
        # if cost < 0.01:
        #     cost = 0

        return {name: float(value) for name, value in terms.items()}
        

    def sample_spindle_update(self, add=None, mtoc_id=None):
        """
        Proposes a single MT attach/detach for optimize()'s Metropolis-Hastings loop.
        Does not mutate any state itself -- it only decides and describes the change;
        optimize() applies (and, on rejection, reverts) it.

        Returns:
            is_disk (bool): True if the target site is on the metaphase plate.
            push (bool): True for a pushing-lattice site, False for a pulling-lattice site.
            front (bool or None): which disk face (True = +e1, False = -e1); None when
                is_disk is False (boundary sites have no face).
            lattice_site (int): index within the relevant state array.
            site_value: mtoc_id to attach, or 0 to detach.

            Returns None instead when there is nothing to propose (no empty reachable site of
            the drawn type, or nothing attached to remove). optimize() counts that as a
            rejected attempt rather than writing a placeholder change into a state array.
        """

        # choose whether to add or remove an MT
        if add is None:
            add = self.rng.choice([True, False], p=[0.5, 0.5]) # True -> add, False -> remove

        if add:

            # choose which mtoc we are nucleating from
            if mtoc_id is None:
                mtoc_id = self.rng.choice(list(self.mtoc_positions.keys()))
            mtoc_position = self.mtoc_positions[mtoc_id]

            # -- 50 / 50 chance of the next MT being a pusher or a puller --
            # drawn *before* the site, so the candidate pool below is built from this type's own
            # state arrays. Drawing the site from the push arrays and flipping the coin afterwards
            # (as this used to) let a puller land on an already-occupied pull site and overwrite
            # it -- a no-op or an MTOC swap rather than a nucleation -- which suppressed puller
            # nucleation in proportion to pull occupancy and broke the push/pull symmetry the
            # proposal is meant to have (any bias between them should come from the cost).
            push = self.rng.choice([True, False], p=[0.5, 0.5]) # True -> push, False -> pull

            if push:
                boundary_state, boundary_lattice = self.push_state, self.push_lattice
                disk_faces = ((True, self.disk_push_state_front), (False, self.disk_push_state_back))
                disk_local = self.disk_push_local
            else:
                boundary_state, boundary_lattice = self.pull_state, self.pull_lattice
                disk_faces = ((True, self.disk_pull_state_front), (False, self.disk_pull_state_back))
                disk_local = self.disk_pull_local

            # -- gather every currently reachable, empty site of this type across the boundary and the disk --
            # each pool segment is (is_disk, front, indices_within_its_own_array), its sites
            # matched 1:1, in order, with that segment's block of pool_positions. Kept as index
            # arrays rather than one (is_disk, front, index) tuple per site: with ~160k empty
            # boundary sites, building those tuples cost more than the rest of the proposal.
            pool_positions = []
            pool_segments = []

            boundary_empty = np.where(boundary_state == 0)[0]
            if boundary_empty.size > 0:
                boundary_positions = boundary_lattice[boundary_empty]
                if self.disk_shadowing_enabled:
                    # drop boundary sites the disk currently blocks from this mtoc's view
                    accessible = self.boundary_sites_accessible(mtoc_id, boundary_positions)
                    boundary_empty = boundary_empty[accessible]
                    boundary_positions = boundary_positions[accessible]
                if boundary_empty.size > 0:
                    pool_positions.append(boundary_positions)
                    pool_segments.append((False, None, boundary_empty))

            # a disk face is only offered as a candidate if this mtoc sits on the side that
            # can actually reach it in a straight line (is_disk_face_reachable)
            for front, disk_state in (() if not self.disk_enabled else disk_faces):
                if not self.is_disk_face_reachable(mtoc_id, front=front):
                    continue
                disk_empty = np.where(disk_state == 0)[0]
                if disk_empty.size == 0:
                    continue
                disk_positions, _, _, _, _ = self._disk_lab_frame(disk_local[disk_empty], 1.0 if front else -1.0)
                pool_positions.append(disk_positions)
                pool_segments.append((True, front, disk_empty))

            if not pool_segments:
                # nothing of this type empty/reachable anywhere for this mtoc right now
                return None

            # # -- exponential length distributed sampling, pooled across every candidate --
            # pool_positions = np.concatenate(pool_positions, axis=0)
            # candidate_distances = normalize_vecs(pool_positions - mtoc_position)[1] # norms of difference vectors
            # length_probability = np.exp(-candidate_distances / self.average_mt_length) # calculate probabilities of an MT growing to be at least that long
            # site_selection_probabilities = length_probability / np.sum(length_probability) # normalize
            #
            # chosen = self.rng.choice(len(pool_positions), p=site_selection_probabilities)

            # -- uniform sampling: every empty, reachable site (boundary or disk) is equally likely --
            num_candidates = sum(len(indices) for _, _, indices in pool_segments)
            chosen = self.rng.integers(num_candidates)
            # walk the segments to the one holding the chosen site
            for is_disk, front, indices in pool_segments:
                if chosen < len(indices):
                    index = indices[chosen]
                    break
                chosen -= len(indices)

            # # -- push or pull: promote to the nearby motor if site_to_fill is within the capture radius --
            # if is_disk:
            #     promote_map = self.disk_push_to_pull_front if front else self.disk_push_to_pull_back
            #     disk_pull_state = self.disk_pull_state_front if front else self.disk_pull_state_back
            #     push = True
            #     if index in promote_map and disk_pull_state[promote_map[index]] == 0:
            #         push = False
            #         index = promote_map[index]
            # else:
            #     push = True
            #     if index in self.push_to_pull and self.pull_state[self.push_to_pull[index]] == 0:
            #         push = False
            #         index = self.push_to_pull[index]

            return is_disk, push, front, index, mtoc_id

        else: # remove
            # choose which pool to remove from, weighted by the proportion of MTs currently in
            # each of the six pools (boundary push/pull, disk push/pull x front/back)
            pools = [
                (False, True, None, self.push_state),
                (False, False, None, self.pull_state),
            ]
            # with the plate disabled the four disk pools are permanently empty; leaving them in
            # would still hand them 4/6 of the draw whenever nothing is attached anywhere, and
            # each such draw is a wasted attempt against the num_attempts budget
            if self.disk_enabled:
                pools += [
                    (True, True, True, self.disk_push_state_front),
                    (True, True, False, self.disk_push_state_back),
                    (True, False, True, self.disk_pull_state_front),
                    (True, False, False, self.disk_pull_state_back),
                ]
            counts = np.array([np.count_nonzero(state) for _, _, _, state in pools], dtype=float)
            total = counts.sum()
            probabilities = np.full(len(pools), 1.0 / len(pools)) if total == 0 else counts / total

            is_disk, push, front, state = pools[self.rng.choice(len(pools), p=probabilities)]

            # choose a filled site to empty, uniformly at random within the chosen pool
            filled_sites = np.where(state != 0)[0]
            if filled_sites.size == 0: # avoiding self.rng errors / nothing to remove in this pool
                return None

            site_to_empty = self.rng.choice(filled_sites)

            return is_disk, push, front, site_to_empty, 0

    def _evaluate_proposal(self, proposal, old_cost):
        """
        One Metropolis-Hastings attempt: applies a proposal from sample_spindle_update,
        integrates forward, culls newly shadowed MTs, costs the result, and decides
        acceptance against old_cost. Leaves the spindle in the proposed state either way;
        _revert_proposal undoes it, which optimize() does on rejection and every
        ProposalPool worker does after every attempt.

        Returns:
            dict: the proposal, whether it was accepted, its cost and culled MTs, the state
            it replaced (for _revert_proposal), and the state it produced (for
            _apply_accepted_change).
        """
        is_disk, push, front, lattice_site, site_value = proposal

        target_state = self._resolve_state_array(is_disk, push, front)
        old_site_value = target_state[lattice_site]
        # time_evolution replaces mtoc_positions and the disk pose rather than mutating them
        # in place, so holding references is enough to restore them
        old_mtoc_positions = self.mtoc_positions
        old_disk_pose = (self.disk_center, self.disk_e1, self.disk_e2, self.disk_e3)

        target_state[lattice_site] = site_value

        new_mtoc_positions, boundary_violated = self.time_evolution()
        self.mtoc_positions = new_mtoc_positions

        # the plate has just moved, so cut off any MT it now blocks before
        # costing this state -- a shadowed MT must stop contributing force and
        # tubulin, and the optimizer should see that consequence of the new pose
        culled = self.cull_shadowed_microtubules()

        # costed through the per-term breakdown so the terms can be saved alongside the total;
        # the sum is calculate_cost()'s, term for term and in the same order
        cost_terms = self.calculate_cost_terms()
        cost = 0
        for name, value in cost_terms.items():
            cost += self.cost_coefficients[name] * value

        # states which violated the boundary are always rejected
        # otherwise, we follow the Metropolis-Hastings condition
        accepted = False
        if cost < old_cost and not boundary_violated:
            accepted = True # accept strict decreases in cost
        elif not boundary_violated:
            # it must be true that cost >= old_cost
            cost_delta = cost - old_cost # > 0
            prob_of_acceptance = np.exp(-cost_delta / self.optimization_temperature)

            accepted = self.rng.choice([True, False], p=[prob_of_acceptance, 1-prob_of_acceptance])

        return {
            'proposal': proposal,
            'accepted': bool(accepted),
            'cost': cost,
            'cost_terms': cost_terms,
            'culled': culled,
            'old_site_value': old_site_value,
            'old_mtoc_positions': old_mtoc_positions,
            'old_disk_pose': old_disk_pose,
            'mtoc_positions': self.mtoc_positions,
            'disk_pose': (self.disk_center, self.disk_e1, self.disk_e2, self.disk_e3),
        }

    def _revert_proposal(self, evaluation):
        """Undoes everything _evaluate_proposal did, given the dict it returned."""
        # unwind in reverse order: the cull ran after the sampled change, and
        # may have cleared that very site (nucleated onto a site the moved plate
        # then blocked). Putting the cull back first lets the single-element
        # restore below have the final say on that site.
        for culled_is_disk, culled_push, culled_front, culled_site, culled_value in evaluation['culled']:
            self._resolve_state_array(culled_is_disk, culled_push, culled_front)[culled_site] = culled_value
        # restore only the single element that was changed
        is_disk, push, front, lattice_site, _ = evaluation['proposal']
        self._resolve_state_array(is_disk, push, front)[lattice_site] = evaluation['old_site_value']
        # reset mtoc positions and disk pose
        self.mtoc_positions = evaluation['old_mtoc_positions']
        self.disk_center, self.disk_e1, self.disk_e2, self.disk_e3 = evaluation['old_disk_pose']

    def _apply_accepted_change(self, change):
        """
        Puts this spindle in the state an accepted proposal produced, from the
        state it was proposed from, without re-integrating it: sets the proposed
        site, clears what the shadow cull cleared, and takes the resulting MTOC
        positions and disk pose. This is how a ProposalPool keeps every copy of
        the spindle identical after each round.
        """
        is_disk, push, front, lattice_site, site_value = change['proposal']
        self._resolve_state_array(is_disk, push, front)[lattice_site] = site_value
        # same order as _evaluate_proposal: the cull ran after the flip, and may clear that same site
        for culled_is_disk, culled_push, culled_front, culled_site, _ in change['culled']:
            self._resolve_state_array(culled_is_disk, culled_push, culled_front)[culled_site] = 0
        self.mtoc_positions = change['mtoc_positions']
        self.disk_center, self.disk_e1, self.disk_e2, self.disk_e3 = change['disk_pose']


    def optimize(self, total_attempts, save_batch_size=1000, max_lab_time=None, callback=None, callback_every=None, num_workers=1, readout_every=2.0):
        """
        Run the optimize loop until self.num_attempts exceeds total_attempts
        or, if max_lab_time (seconds of simulated/lab time) is given, until
        self.time exceeds it -- whichever happens first.

        If callback is given, callback(self) is called every time
        self.num_accepted_states reaches a multiple of callback_every (e.g. to
        make plots mid-run). Pending accepted states are flushed to disk first,
        so anything the callback reads back from the traces is complete. With
        num_workers > 1 every worker waits on the callback, so prefer
        plot_progress.py run as its own job.

        num_workers > 1 evaluates that many proposals at once in forked worker
        processes (see ProposalPool), with the same statistics as the serial
        loop. Linux only (it relies on fork); size it to the cores SLURM gives
        the job, e.g. int(os.environ['SLURM_CPUS_PER_TASK']).

        The live readout table is redrawn at most every readout_every seconds
        of wall time.
        """

        initial_cost = len(self.push_state) # setting initial cost to be very high
        old_mtoc_positions = self.mtoc_positions.copy() # initial original state is the current state
        old_disk_center = self.disk_center.copy()
        old_disk_e1, old_disk_e2, old_disk_e3 = self.disk_e1.copy(), self.disk_e2.copy(), self.disk_e3.copy()
        old_cost = initial_cost # any stable position is an improvement
        old_time = np.copy(self.time)
        # trace will be saved every save_batch_size accepted states.
        # the spindle_trace tracks changes to the boundary; disk_spindle_trace is a separate,
        # parallel log for changes to the metaphase plate (see multi_aster_spindle.py plan notes)
        spindle_trace = []
        disk_spindle_trace = []
        # timepoints added to self.trajectory since the last flush; written out as one chunk
        trajectory_batch = {}

        # # accepted_states is a list where each element is the tuple (push_state, pull_state)
        # accepted_pull_states = []
        # accepted_push_states = []
        num_accepted_states_at_empty = np.copy(self.num_accepted_states)

        def flush_accepted_states():
            """Persist the accumulated accepted states as a batch and reset the buffers."""
            # nonlocal accepted_push_states, accepted_pull_states, num_accepted_states_at_empty
            nonlocal spindle_trace, disk_spindle_trace, trajectory_batch, num_accepted_states_at_empty
            if not spindle_trace and not disk_spindle_trace and not trajectory_batch:
                return

            start = num_accepted_states_at_empty
            end = self.num_accepted_states
            if spindle_trace:
                safe_np_save(np.array(spindle_trace), os.path.join(self.spindle_trace_path, f'spindle_trace_{start}_{end}.npy' ))
            if disk_spindle_trace:
                safe_np_save(np.array(disk_spindle_trace), os.path.join(self.spindle_trace_path, f'disk_trace_{start}_{end}.npy'))
            if trajectory_batch:
                safe_pickle_dump(trajectory_batch, os.path.join(self.trajectory_path, f'trajectory_{start}_{end}.pkl'))
            # np.save(os.path.join(self.spindle_trace_path, trace_batch_name('push', start, end)), np.array(accepted_push_states))
            # np.save(os.path.join(self.spindle_trace_path, trace_batch_name('pull', start, end)), np.array(accepted_pull_states))

            # accepted_push_states = []
            # accepted_pull_states = []
            spindle_trace = []
            disk_spindle_trace = []
            trajectory_batch = {}
            num_accepted_states_at_empty = np.copy(self.num_accepted_states)

            # keep the experiment-level counters in sync with what's on disk
            safe_np_save(self.num_accepted_states, os.path.join(self.dir_path, 'num_accepted_states.npy'))
            safe_np_save(self.num_attempts, os.path.join(self.dir_path, 'num_attempts.npy'))


        # -- wall time readout --
        start = time.perf_counter()
        last_readout = -np.inf # so the first pass draws the table

        def format_elapsed(start):
            elapsed = time.perf_counter() - start
            return f"{elapsed:.2f}s"

        def lab_time_exceeded():
            return max_lab_time is not None and self.time >= max_lab_time

        def keep_going():
            return self.num_attempts <= total_attempts and not lab_time_exceeded()

        # forked before Live starts its refresh thread; closed (workers stopped) however the loop exits
        pool_context = ProposalPool(self, num_workers) if num_workers > 1 else contextlib.nullcontext()

        with Live(console=console, refresh_per_second=4) as live, pool_context as pool:

            # --- update spindle -> relax loop ---

            while keep_going():

                # -- middle loop (update spindle, try to relax, if cost improves, accept, if not, reject) --
                acceptable = False
                attempt_counter = 0

                while not acceptable and keep_going():

                    if pool is not None:
                        # set the workers going first, so the readout below renders while they compute
                        pool.dispatch(old_cost)

                    # -- readout --
                    # rebuilt every readout_every seconds of wall time rather than every attempt:
                    # several rows (forces, tubulin use, MT counts) rescan the whole lattice
                    if time.perf_counter() - last_readout >= readout_every:
                        last_readout = time.perf_counter()
                        # initialize table
                        outer_table = Table(title="Spindle Simulation")
                        outer_table.add_column("Parameter", justify="left")
                        outer_table.add_column("Value", justify="right")
                    
                        # set table values
                        outer_table.add_row('Wall elapsed time (s)', format_elapsed(start))
                        if max_lab_time is not None:
                            outer_table.add_row('Last Accepted Time (s)', f"{old_time} / {max_lab_time:.2f}s") # last stable time
                        else:
                            outer_table.add_row('Last Accepted Time (s)', str(old_time)) # last stable time
                        outer_table.add_row('Last Accepted Position (um)', str(old_mtoc_positions)) # last stable position
                        # last stable spindle centre (MTOC centroid) -- the point the centring term in calculate_cost() targets
                        old_spindle_center = np.mean(list(old_mtoc_positions.values()), axis=0)
                        outer_table.add_row('Spindle Centre (um)', str(old_spindle_center))
                        if self.disk_enabled:
                            outer_table.add_row('Disk Centre (um)', str(old_disk_center)) # last stable disk position
                            outer_table.add_row('Spindle Centre - Disk Centre Distance (um)', f'{normalize_vecs(old_spindle_center - old_disk_center)[1]:.4f}')
                            outer_table.add_row('Disk Normal (e1)', str(old_disk_e1)) # last stable disk orientation
                            # the same quantities the disk velocity cost terms penalize, in the last stable state
                            disk_velocity, disk_omega = self.calculate_disk_velocity_and_omega()
                            outer_table.add_row('Disk COM Velocity (um/s)', f'{disk_velocity}  |U| = {np.linalg.norm(disk_velocity):.4g}')
                            outer_table.add_row('Disk Angular Velocity (rad/s)', f'{disk_omega}  |omega| = {np.linalg.norm(disk_omega):.4g}')
                            # only the part of omega perpendicular to e1 moves the normal; spin about e1 doesn't
                            outer_table.add_row('Disk Normal Tumble Rate (rad/s)', f'{np.linalg.norm(np.cross(disk_omega, self.disk_e1)):.4g}')
                        if len(list(self.mtoc_positions.keys())) == 2:
                            # last stable spindle orientation, MTOC 1 -> MTOC 2, same axis as in calculate_cost()
                            old_spindle_axis = normalize_vecs(old_mtoc_positions[2] - old_mtoc_positions[1])[0]
                            outer_table.add_row('Spindle Axis (MTOC 1 -> 2)', str(old_spindle_axis))
                            if self.disk_enabled:
                                # angle to the nearest of +/-e1, in [0, 90] deg -- the quantity squared in the alignment cost
                                old_misalignment = np.degrees(np.arccos(np.clip(np.abs(np.dot(old_spindle_axis, old_disk_e1)), 0.0, 1.0)))
                                outer_table.add_row('Spindle Axis / Disk Normal Angle (deg)', f'{old_misalignment:.2f}')
                            outer_table.add_row('Spindle Length (um)', f'{normalize_vecs(self.mtoc_positions[1] - self.mtoc_positions[2])[1]} / {self.spindle_length}') # tubulin use / tubulin budget
                        outer_table.add_row('Net Force on aster 1 (pN)', f'{self.calc_mtoc_velocity(1)*self.cytoplasmic_drag_factor}') # force vector on MTOC
                        outer_table.add_row('Last Accepted Cost', str(old_cost)) # last accepted cost
                        outer_table.add_row('Total number of MTs', str(self.calculate_num_mts())) # cortical + chromosomal, push + pull
                        outer_table.add_row('Number of cortical pushing MTs', str(np.count_nonzero(self.push_state))) # boundary push sites
                        outer_table.add_row('Number of cortical pulling MTs', str(np.count_nonzero(self.pull_state))) # boundary pull sites
                        if self.disk_enabled:
                            outer_table.add_row('Number of chromosomal pushing MTs', str(np.count_nonzero(self.disk_push_state_front) + np.count_nonzero(self.disk_push_state_back))) # disk push sites, both faces
                            outer_table.add_row('Number of chromosomal pulling MTs', str(np.count_nonzero(self.disk_pull_state_front) + np.count_nonzero(self.disk_pull_state_back))) # disk pull sites, both faces
                        outer_table.add_row('Tubulin use / tubulin budget (um)', f'{self.calculate_tubulin_use()} / {self.tubulin_budget}') # tubulin use / tubulin budget
                        outer_table.add_row('Attempt Counter', str(attempt_counter)) # attempt counter
                        outer_table.add_row('Number of total attempts', f"{self.num_attempts} / {total_attempts}")
                        outer_table.add_row('Number of Positions Accepted', str(self.num_accepted_states)) # number of accepted positions
                        proc_rss_gb = psutil.Process().memory_info().rss / 1e9
                        slurm_mb = os.environ.get("SLURM_MEM_PER_NODE")
                        mem_str = f"{proc_rss_gb:.2f} GB"
                        if slurm_mb:
                            mem_str += f" / {int(slurm_mb)/1e3:.1f} GB alloc"
                        outer_table.add_row('Memory (RSS)', mem_str)
                        live.update(outer_table)

                    if pool is not None:
                        # a whole round of num_workers proposals; see ProposalPool for why the
                        # winner is the lowest-numbered accepted one
                        winner, num_tried = pool.collect()
                        attempt_counter += num_tried
                        self.num_attempts += num_tried
                        if winner is None:
                            continue

                        self._apply_accepted_change(winner)
                        is_disk, push, front, lattice_site, site_value = winner['proposal']
                        culled = winner['culled']
                        meta_cost = winner['cost']
                        meta_cost_terms = winner['cost_terms']
                        meta_time = self.time + self.evolution_time
                        acceptable = True
                        continue

                    # update meta-spindle-state
                    proposal = self.sample_spindle_update()

                    if proposal is None:
                        # nothing to propose: a rejected attempt, the state stays put. The old
                        # placeholder (False, True, None, 0, 0) wrote 0 into push_state[0] here,
                        # silently detaching any pusher at boundary site 0.
                        attempt_counter += 1
                        self.num_attempts += 1
                        continue

                    attempt_counter += 1
                    self.num_attempts +=1

                    evaluation = self._evaluate_proposal(proposal, old_cost)
                    acceptable = evaluation['accepted']

                    # reverse metastate if the spindle modification is not acceptable
                    if not acceptable:
                        self._revert_proposal(evaluation)

                    is_disk, push, front, lattice_site, site_value = proposal
                    culled = evaluation['culled']
                    meta_cost = evaluation['cost']
                    meta_cost_terms = evaluation['cost_terms']
                    meta_time = self.time + self.evolution_time

                # the inner loop also exits when the attempt/lab-time budget runs out with
                # nothing accepted. That last state was rejected and fully reverted, so it is
                # not a new accepted state: committing it here would log the rejected change
                # (and the shadow cull that went with it) to the trace as though it happened,
                # leaving a replay one site out of step with the run.
                if not acceptable:
                    break

                # -- back in outer loop, saving new accepted position --
                # current mtoc positions and spindle states become old mtoc positions and spindle states
                old_mtoc_positions = self.mtoc_positions.copy() # this is the current metastate position
                old_disk_center = self.disk_center.copy()
                old_disk_e1, old_disk_e2, old_disk_e3 = self.disk_e1.copy(), self.disk_e2.copy(), self.disk_e3.copy()
                old_cost = meta_cost
                old_time = meta_time
                self.time = meta_time
                if self.save:
                    if is_disk:
                        disk_spindle_trace.append((push, front, lattice_site, site_value))
                    else:
                        spindle_trace.append((push, lattice_site, site_value))
                    # the shadow cull is part of this accepted state, so it has to reach the
                    # traces too -- otherwise a replay (occupancy/lifetime plots) keeps MTs the
                    # run itself cut off
                    for culled_is_disk, culled_push, culled_front, culled_site, _ in culled:
                        if culled_is_disk:
                            disk_spindle_trace.append((culled_push, culled_front, culled_site, 0))
                        else:
                            spindle_trace.append((culled_push, culled_site, 0))

                self.num_accepted_states += 1
                attempt_counter = 0

                # fold in trajectory. This is an empty dictionary if self.save_trajectory == False
                # self.trajectory = self.trajectory | meta_trajectory
                if self.save_trajectory:
                    total_force = {}
                    for mtoc_id in self.mtoc_positions.keys():
                        total_force[mtoc_id] = self.cytoplasmic_drag_factor * self.calc_mtoc_velocity(mtoc_id)
                    tubulin_use_per_mtoc = self.calculate_tubulin_use_per_mtoc()
                    timepoint_data = {
                        # 'push_state': self.push_state.astype(int),
                        # 'pull_state': self.pull_state.astype(int),
                        'mtoc_pos': self.mtoc_positions,
                        'cost': meta_cost, # this very state's cost, already computed when it was proposed
                        'cost_terms': meta_cost_terms, # its terms before their prefactors (see calculate_cost_terms)
                        'tubulin_use': sum(tubulin_use_per_mtoc.values()),
                        'tubulin_use_per_mtoc': tubulin_use_per_mtoc,
                        'num_mts': self.calculate_num_mts(),
                        'total_force': total_force,
                    }
                    # metaphase plate (disk) snapshot, for animate_mtoc_trajectory. Position/
                    # orientation change every accepted step (time_evolution's EOM); occupancy
                    # changes whenever sample_spindle_update picks a disk site this step.
                    # Omitted when the plate is disabled -- it never moves and holds nothing, so
                    # the keys would be dead weight on every accepted step.
                    if self.disk_enabled:
                        timepoint_data.update({
                            'disk_center': self.disk_center.copy(),
                            'disk_e1': self.disk_e1.copy(),
                            'disk_e2': self.disk_e2.copy(),
                            'disk_e3': self.disk_e3.copy(),
                            'disk_push_state_front': self.disk_push_state_front.copy(),
                            'disk_push_state_back': self.disk_push_state_back.copy(),
                            'disk_pull_state_front': self.disk_pull_state_front.copy(),
                            'disk_pull_state_back': self.disk_pull_state_back.copy(),
                        })
                    self.trajectory[self.time] = timepoint_data
                    trajectory_batch[self.time] = timepoint_data


                if self.save and (self.num_accepted_states - num_accepted_states_at_empty) >= save_batch_size:
                    # save the last batch of accepted states to the spindle_trace folder
                    # trajectory is only used for making animations.
                    flush_accepted_states()

                if callback is not None and callback_every and self.num_accepted_states % callback_every == 0:
                    if self.save:
                        flush_accepted_states()
                    callback(self)

        # persist any trailing accepted states that didn't fill a final 1000-state batch
        # if self.save:
        #     flush_accepted_states()

    def plot_cost(self, start_time=None, end_time=None):
        """
        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
        """
        times = list(self.trajectory.keys())
        if start_time is not None:
            times = [t for t in times if t >= start_time]
        if end_time is not None:
            times = [t for t in times if t <= end_time]

        costs = []
        for time in times:
            costs.append(self.trajectory[time]['cost'])

        plt.figure(figsize=(6, 3))
        plt.plot(times, costs)
        plt.xlabel('time (s)')
        plt.ylabel('cost (dimensionless)')
        # plt.title(f'2_aster_{self.tubulin_budget}_tubulin_{self.optimization_temperature}_temp')
        # plt.show()
        plt.tight_layout()
        cost_plot_path = os.path.join(self.plot_folder_path, f'cost_time_plot.png')
        plt.savefig(cost_plot_path)
        print(f'cost vs time plot saved to {cost_plot_path}')
        plt.close()

    def plot_cost_terms(self, start_time=None, end_time=None, save_path=None):
        """
        Each calculate_cost term vs time, one panel per term. The left axis is the term's
        dimensionless value before its prefactor; the right axis (only when the prefactor is
        nonzero) is its contribution to the cost, prefactor * value. A term with a zero
        prefactor is still drawn -- it just doesn't enter the cost.

        Uses the 'cost_terms' recorded with each accepted state, so timepoints saved before
        that was recorded are skipped.

        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
            save_path (str): if given, write the figure here instead of plot_folder_path
        """
        times = [t for t in self.trajectory.keys() if 'cost_terms' in self.trajectory[t]]
        if start_time is not None:
            times = [t for t in times if t >= start_time]
        if end_time is not None:
            times = [t for t in times if t <= end_time]
        if not times:
            print('no timepoints with recorded cost_terms -- skipping plot_cost_terms')
            return

        recorded = set().union(*(self.trajectory[t]['cost_terms'].keys() for t in times))
        names = [name for name in self.cost_coefficients if name in recorded]

        fig, axes = plt.subplots(len(names), 1, sharex=True, figsize=(7, 1.9 * len(names)), squeeze=False)
        for ax, name in zip(axes[:, 0], names):
            values = [self.trajectory[t]['cost_terms'].get(name, np.nan) for t in times]
            coefficient = self.cost_coefficients[name]
            ax.plot(times, values)
            ax.set_ylabel('term', fontsize=8)
            if coefficient != 0:
                ax.set_title(f'{name}  (prefactor {coefficient:g})', fontsize=9)
                contribution_axis = ax.secondary_yaxis('right', functions=(lambda x, c=coefficient: c * x,
                                                                            lambda y, c=coefficient: y / c))
                contribution_axis.set_ylabel('× prefactor', fontsize=8)
            else:
                ax.set_title(f'{name}  (prefactor 0 -- not in cost)', fontsize=9)
        axes[-1, 0].set_xlabel('time (s)')
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'cost_terms_time_plot.png')
        plt.savefig(save_path)
        plt.close()
        print(f'cost terms vs time plot saved to {save_path}')



    def _reconstruct_tubulin_use_per_mtoc(self, times):
        """
        Per-MTOC MT length at the given trajectory times, for timepoints saved before
        'tubulin_use_per_mtoc' was recorded. Boundary occupancy comes from replaying the
        spindle_trace up to each time; MTOC positions, and the disk's pose and occupancy,
        come from the trajectory entry itself.

        Args:
            times (list[float]): trajectory keys, in increasing order.

        Returns:
            dict[float, dict[int, float]]: time -> (mtoc_id -> total MT length (um))
        """
        mtoc_ids = sorted(self.mtoc_positions.keys())
        push_state = np.zeros(self.num_push_sites)
        pull_state = np.zeros(self.num_pull_sites)
        trace = self._replay_trace('spindle_trace')
        pending = next(trace, None)

        lengths = {}
        for t in times:
            # a trace row with global index i is the change made by accepted state i + 1,
            # i.e. the state the clock reads (i + 1) * evolution_time after
            num_states = int(np.round(t / self.evolution_time))
            while pending is not None and pending[0] < num_states:
                push, lattice_site, site_value = pending[1]
                if push:
                    push_state[int(lattice_site)] = site_value
                else:
                    pull_state[int(lattice_site)] = site_value
                pending = next(trace, None)

            timepoint = self.trajectory[t]
            mtoc_pos = timepoint['mtoc_pos']
            lengths[t] = {}
            for mtoc_id in mtoc_ids:
                mt_length = np.sum(normalize_vecs(self.push_lattice[push_state == mtoc_id] - mtoc_pos[mtoc_id])[1])
                mt_length += np.sum(normalize_vecs(self.pull_lattice[pull_state == mtoc_id] - mtoc_pos[mtoc_id])[1])

                if 'disk_center' in timepoint:
                    for key, local in [
                        ('disk_push_state_front', self.disk_push_local),
                        ('disk_push_state_back', self.disk_push_local),
                        ('disk_pull_state_front', self.disk_pull_local),
                        ('disk_pull_state_back', self.disk_pull_local),
                    ]:
                        connected = timepoint[key] == mtoc_id
                        if not connected.any():
                            continue
                        # both faces share the same in-plane site positions (see _disk_lab_frame)
                        lab_pos = (timepoint['disk_center'] + local[connected, 0:1] * timepoint['disk_e2']
                                   + local[connected, 1:2] * timepoint['disk_e3'])
                        mt_length += np.sum(normalize_vecs(lab_pos - mtoc_pos[mtoc_id])[1])

                lengths[t][mtoc_id] = float(mt_length)

        return lengths

    def plot_mt_length(self, save_path=None, start_time=None, end_time=None, max_reconstructed_points=2000):
        """
        Plots the total length of all MTs, and the total length of the MTs coming from
        each MTOC (boundary and disk MTs combined), against time. The tubulin budget is
        drawn as a dashed line for reference.

        Per-MTOC lengths are read from the trajectory's 'tubulin_use_per_mtoc' where it
        was recorded; timepoints saved before that field existed are reconstructed by
        replaying the spindle_trace (see _reconstruct_tubulin_use_per_mtoc), subsampled
        to at most max_reconstructed_points timepoints.

        Args:
            save_path (str): where to save the plot. Defaults to plot_folder_path/mt_length.png
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
            max_reconstructed_points (int): cap on the number of unrecorded timepoints
                whose per-MTOC lengths are reconstructed from the trace
        """
        times = list(self.trajectory.keys())
        if start_time is not None:
            times = [t for t in times if t >= start_time]
        if end_time is not None:
            times = [t for t in times if t <= end_time]

        total_lengths = [self.trajectory[t]['tubulin_use'] for t in times]

        mtoc_ids = sorted(self.mtoc_positions.keys())
        recorded = {t: self.trajectory[t]['tubulin_use_per_mtoc'] for t in times if 'tubulin_use_per_mtoc' in self.trajectory[t]}
        missing = [t for t in times if t not in recorded]
        if missing:
            if len(missing) > max_reconstructed_points:
                keep = np.linspace(0, len(missing) - 1, max_reconstructed_points).astype(int)
                missing = [missing[i] for i in keep]
            recorded.update(self._reconstruct_tubulin_use_per_mtoc(missing))
        per_mtoc_times = [t for t in times if t in recorded]
        per_mtoc_lengths = {mtoc_id: [recorded[t].get(mtoc_id, 0.0) for t in per_mtoc_times]
                            for mtoc_id in mtoc_ids}

        # num_mtocs = len(mtoc_ids)
        # shades = np.linspace(0.4, 0.9, num_mtocs) if num_mtocs > 1 else np.array([0.7])
        # mtoc_colors = plt.cm.Greens(shades)
        mtoc_colors = ['tab:orange', 'tab:blue']

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(times, total_lengths, color='k', label='total')
        if per_mtoc_times:
            for i, mtoc_id in enumerate(mtoc_ids):
                ax.plot(per_mtoc_times, per_mtoc_lengths[mtoc_id], color=mtoc_colors[i], label=f'MTOC {mtoc_id}')
        # ax.axhline(self.tubulin_budget, color='gray', linestyle='--', linewidth=1, label='tubulin budget')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'total MT length ($\mu$m)')
        ax.legend()
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'mt_length.png')
        plt.savefig(save_path)
        plt.close()
        print(f'MT length plot saved to {save_path}')

        return times, total_lengths, per_mtoc_times, per_mtoc_lengths

    
    def plot_aster_separation(self, start_time=None, end_time=None):
        """
        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
        """
        times = list(self.trajectory.keys())
        if start_time is not None:
            times = [t for t in times if t >= start_time]
        if end_time is not None:
            times = [t for t in times if t <= end_time]

        norm1 = []
        norm2 = []
        aster_separations = []
        for time in times:
            mtoc_positions = self.trajectory[time]['mtoc_pos']
            norm1.append(normalize_vecs(mtoc_positions[1])[1])
            norm2.append(normalize_vecs(mtoc_positions[2])[1])
            aster_separations.append(normalize_vecs(mtoc_positions[1] - mtoc_positions[2])[1])


        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        fig, ax2 = plt.subplots(figsize=(6, 3))

        # plotting aster norms
        # ax1.plot(times, norm1, label='aster 1 distance from centre')
        # ax1.plot(times, norm2, label='aster 2 distance from centre')
        # ax1.set_ylabel('Distance (um)')
        # ax1.set_ylim([0, self.boundary_radius])
        # ax1.legend()

        # plotting separation
        ax2.plot(times, aster_separations, label='distance between asters')
        ax2.set_ylim([0, 2* self.boundary_radius])
        ax2.set_ylabel('aster separation (um)')
        ax2.set_xlabel('time (s)')

        # plotting spindle deviation from origin (R1-R2)

        fig.tight_layout()

        # aster_norm_plot_path = os.path.join(spindle.plot_folder_path, f'aster_norms.png')
        # aster_sep_plot_path = os.path.join(spindle.plot_folder_path, f'aster_separation.png')
        combined_plot_path = os.path.join(self.plot_folder_path, f'aster_separation.png')
        plt.savefig(combined_plot_path)
        plt.close()
        print(f'aster separation saved to {combined_plot_path}')

    def plot_aster_distance_from_centre(self, start_time=None, end_time=None):
        """
        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
        """
        times = list(self.trajectory.keys())
        if start_time is not None:
            times = [t for t in times if t >= start_time]
        if end_time is not None:
            times = [t for t in times if t <= end_time]
        mtoc_ids = sorted(self.trajectory[times[0]]['mtoc_pos'].keys())
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        distances = {mtoc_id: [] for mtoc_id in mtoc_ids}
        for time in times:
            mtoc_positions = self.trajectory[time]['mtoc_pos']
            for mtoc_id in mtoc_ids:
                distances[mtoc_id].append(normalize_vecs(mtoc_positions[mtoc_id])[1])

        fig, ax = plt.subplots(figsize=(6, 3))
        for i, mtoc_id in enumerate(mtoc_ids):
            ax.plot(times, distances[mtoc_id], color=colors[i % len(colors)], label=f'aster {mtoc_id} distance from centre\n mean={np.round(np.mean(distances[mtoc_id]), 3)} +/- {np.round(np.std(distances[mtoc_id]), 3)}')
        ax.set_ylabel('Distance (um)')
        ax.set_ylim([0, self.boundary_radius])
        ax.set_xlabel('time (s)')
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(self.plot_folder_path, 'aster_distance_from_centre.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'aster distance from centre saved to {plot_path}')

    def _disk_pose_history(self, start_time=None, end_time=None):
        """
        Pull the recorded disk pose out of self.trajectory.

        Returns:
            times (list[float]), centers (N, 3) array, normals (N, 3) array of e1.
        Raises:
            ValueError: if the plate is disabled, the trajectory is empty, or the trajectory
                predates disk pose tracking.
        """
        if not self.disk_enabled:
            raise ValueError('this spindle was built with disk_enabled=False, so it has no disk pose history')

        times = list(self.trajectory.keys())
        if start_time is not None:
            times = [t for t in times if t >= start_time]
        if end_time is not None:
            times = [t for t in times if t <= end_time]
        if not times:
            raise ValueError('no trajectory timepoints in the requested time window')
        if 'disk_center' not in self.trajectory[times[0]]:
            raise ValueError('this trajectory has no recorded disk pose (it predates disk tracking)')

        centers = np.array([self.trajectory[t]['disk_center'] for t in times], dtype=float)
        normals = np.array([self.trajectory[t]['disk_e1'] for t in times], dtype=float)
        return times, centers, normals

    def plot_disk_distance_from_centre(self, start_time=None, end_time=None, save_path=None):
        """
        Distance of the disk centre of mass from the origin, |R_D|, vs time.

        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
            save_path (str): if given, write the figure here instead of plot_folder_path

        Returns:
            None. No-op (with a printed note) when the plate is disabled, so a driver script
            can call the whole plotting suite unconditionally.
        """
        if not self.disk_enabled:
            print('disk_enabled=False -- skipping plot_disk_distance_from_centre')
            return

        times, centers, _ = self._disk_pose_history(start_time, end_time)
        distances = np.linalg.norm(centers, axis=1)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(times, distances, color='tab:purple',
                label=f'disk distance from centre\n mean={np.round(np.mean(distances), 3)} +/- {np.round(np.std(distances), 3)}')
        ax.set_ylabel('|R_D| (um)')
        ax.set_ylim([0, self.boundary_radius])
        ax.set_xlabel('time (s)')
        ax.legend()
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'disk_distance_from_centre.png')
        plt.savefig(save_path)
        plt.close()
        print(f'disk distance from centre saved to {save_path}')

    def plot_disk_orientation(self, start_time=None, end_time=None, save_path=None, unwrap_phi=False):
        """
        Spherical angles of the disk unit normal e1 vs time, using the standard
        convention theta = arccos(e1_z) in [0, pi] (inclination from +z) and
        phi = arctan2(e1_y, e1_x) in (-pi, pi] (azimuth in the xy-plane).

        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
            save_path (str): if given, write the figure here instead of plot_folder_path
            unwrap_phi (bool): if True, unwrap phi so the trace is continuous
                instead of jumping by 2*pi at the +/-pi branch cut

        Returns:
            None. No-op (with a printed note) when the plate is disabled.
        """
        if not self.disk_enabled:
            print('disk_enabled=False -- skipping plot_disk_orientation')
            return

        times, _, normals = self._disk_pose_history(start_time, end_time)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        theta = np.arccos(np.clip(normals[:, 2], -1.0, 1.0))
        phi = np.arctan2(normals[:, 1], normals[:, 0])
        if unwrap_phi:
            phi = np.unwrap(phi)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(times, theta, color='tab:blue', label='theta (inclination from +z)')
        ax.plot(times, phi, color='tab:orange', label='phi (azimuth in xy-plane)')
        ax.set_ylabel('angle (rad)')
        ax.set_xlabel('time (s)')
        if not unwrap_phi:
            ax.set_ylim([-np.pi - 0.1, np.pi + 0.1])
            ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        ax.legend()
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'disk_orientation.png')
        plt.savefig(save_path)
        plt.close()
        print(f'disk orientation saved to {save_path}')

    def plot_spindle_orientation(self, start_time=None, end_time=None, save_path=None, unwrap_phi=False):
        """
        Spherical angles of the spindle axis unit vector (R_1 - R_2)/|R_1 - R_2| vs time,
        using the same convention as plot_disk_orientation: theta = arccos(u_z) in [0, pi]
        (inclination from +z) and phi = arctan2(u_y, u_x) in (-pi, pi] (azimuth in the xy-plane).
        Times where the two MTOCs coincide (no defined axis) are left as gaps.

        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
            save_path (str): if given, write the figure here instead of plot_folder_path
            unwrap_phi (bool): if True, unwrap phi so the trace is continuous
                instead of jumping by 2*pi at the +/-pi branch cut

        Returns:
            None. No-op (with a printed note) unless MTOCs 1 and 2 both exist.
        """
        if 1 not in self.mtoc_positions or 2 not in self.mtoc_positions:
            print('spindle orientation needs MTOCs 1 and 2 -- skipping plot_spindle_orientation')
            return

        times = list(self.trajectory.keys())
        if start_time is not None:
            times = [t for t in times if t >= start_time]
        if end_time is not None:
            times = [t for t in times if t <= end_time]

        axes = np.array([self.trajectory[t]['mtoc_pos'][1] - self.trajectory[t]['mtoc_pos'][2] for t in times], dtype=float)
        lengths = np.linalg.norm(axes, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            axes = axes / lengths[:, None] # coincident MTOCs -> nan, drawn as a gap

        theta = np.arccos(np.clip(axes[:, 2], -1.0, 1.0))
        phi = np.arctan2(axes[:, 1], axes[:, 0])
        if unwrap_phi:
            valid = np.isfinite(phi)
            phi[valid] = np.unwrap(phi[valid])

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(times, theta, color='tab:blue', label='theta (inclination from +z)')
        ax.plot(times, phi, color='tab:orange', label='phi (azimuth in xy-plane)')
        ax.set_title(r'spindle axis $R_1 - R_2$')
        ax.set_ylabel('angle (rad)')
        ax.set_xlabel('time (s)')
        if not unwrap_phi:
            ax.set_ylim([-np.pi - 0.1, np.pi + 0.1])
            ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        ax.legend()
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'spindle_orientation.png')
        plt.savefig(save_path)
        plt.close()
        print(f'spindle orientation saved to {save_path}')

    def plot_surface_occupancy(self, mtoc_id, start_time, end_time, show_occupancy='pull'):
        """
        show_occupancy: one of 'pull', 'push', 'both' (True is treated as 'both').
        Controls which motor occupancy overlays are plotted.
        """
        if show_occupancy is True:
            show_occupancy = 'both'
        show_occupancy = show_occupancy.lower()
        if show_occupancy not in ('pull', 'push', 'both'):
            raise ValueError("show_occupancy must be one of 'pull', 'push', 'both'/True")
        show_pull = show_occupancy in ('pull', 'both')
        show_push = show_occupancy in ('push', 'both')

        fig = plt.figure()
        # computed_zorder=False makes draw order (not true 3D depth) determine what's on top,
        # so the pull scatter (drawn last) stays visible over the far denser push scatter.
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        plt.title(f'occupancy proportion for MTOC {mtoc_id}\nfrom {start_time} to {end_time} seconds')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('z (um)')

        if show_push:
            push_occupancy = self.calculate_push_occupancy(mtoc_id, start_time, end_time)
            push_lattice = self.push_lattice
            push_norm = mcolors.Normalize(vmin=push_occupancy.min(), vmax=push_occupancy.max())
            push_cmap = plt.cm.viridis
            push_rgba = push_cmap(push_norm(push_occupancy))
            # scale alpha with occupancy itself (rather than an occupied/unoccupied cutoff) so
            # lightly-occupied sites fade toward invisible instead of forming an opaque low-value shell
            push_rgba[:, 3] = push_norm(push_occupancy) * 0.5
            ax.scatter(push_lattice[:,0], push_lattice[:,1], push_lattice[:,2],
                       c=push_rgba, zorder=1)
            mappable = plt.cm.ScalarMappable(cmap=push_cmap, norm=push_norm)
            mappable.set_array(push_occupancy)
            fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.15, label="Push occupancy")

        if show_pull:
            pull_occupancy = self.calculate_motor_occupancy(mtoc_id, start_time, end_time)
            pull_lattice = self.pull_lattice
            pull_norm = mcolors.Normalize(vmin=pull_occupancy.min(), vmax=pull_occupancy.max())
            pull_cmap = plt.cm.inferno
            ax.scatter(pull_lattice[:,0], pull_lattice[:,1], pull_lattice[:,2],
                       c=pull_occupancy, alpha=0.7, cmap=pull_cmap, norm=pull_norm, zorder=2)
            mappable = plt.cm.ScalarMappable(cmap=pull_cmap, norm=pull_norm)
            mappable.set_array(pull_occupancy)
            fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1, label="Pull occupancy")

        ax.axis('equal')
        # plt.show()

        plot_path = os.path.join(self.plot_folder_path, f'surface_occupancy_{show_occupancy}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'surface occupancy saved to {plot_path}')

    def plot_occupancy_vs_angle(self, mtoc_id, start_time, end_time, show_occupancy='pull', num_bins=32):
        """
        Plots a histogram of mean occupancy proportion by polar angle (radians) from the axis
        running through the sphere center and the surface point closest to the given MTOC
        (i.e. the axis along the MTOC's own position vector).

        show_occupancy: one of 'pull', 'push', 'both' (True is treated as 'both').
        """
        if show_occupancy is True:
            show_occupancy = 'both'
        show_occupancy = show_occupancy.lower()
        if show_occupancy not in ('pull', 'push', 'both'):
            raise ValueError("show_occupancy must be one of 'pull', 'push', 'both'/True")
        show_pull = show_occupancy in ('pull', 'both')
        show_push = show_occupancy in ('push', 'both')

        axis, _ = normalize_vecs(self.mtoc_positions[mtoc_id])

        def _angles_rad(lattice):
            unit_lattice, _ = normalize_vecs(lattice)
            cos_angle = np.clip(unit_lattice @ axis, -1.0, 1.0)
            return np.arccos(cos_angle)

        bin_edges = np.linspace(0, np.pi, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = np.pi / num_bins

        def _binned_mean(angles, occupancy):
            bin_idx = np.clip(np.digitize(angles, bin_edges) - 1, 0, num_bins - 1)
            means = np.zeros(num_bins)
            for i in range(num_bins):
                mask = bin_idx == i
                if np.any(mask):
                    means[i] = occupancy[mask].mean()
            return means

        fig, ax = plt.subplots()
        plt.title(f'occupancy vs angle from MTOC {mtoc_id} axis, {start_time} to {end_time} seconds')

        if show_pull:
            pull_occupancy = self.calculate_motor_occupancy(mtoc_id, start_time, end_time)
            pull_angles = _angles_rad(self.pull_lattice)
            pull_mean = _binned_mean(pull_angles, pull_occupancy)
            ax.bar(bin_centers, pull_mean, width=bin_width, alpha=0.6,
                   color='tab:red', edgecolor='tab:red', label='Pull occupancy')

        if show_push:
            push_occupancy = self.calculate_push_occupancy(mtoc_id, start_time, end_time)
            push_angles = _angles_rad(self.push_lattice)
            push_mean = _binned_mean(push_angles, push_occupancy)
            ax.bar(bin_centers, push_mean, width=bin_width, alpha=0.6,
                   color='tab:blue', edgecolor='tab:blue', label='Push occupancy')

        ax.set_xlim(0, np.pi)
        ax.set_xlabel('angle from MTOC axis (radians)')
        ax.set_ylabel('occupancy proportion')
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(self.plot_folder_path, f'occupancy_vs_angle.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'occupancy vs angle plot saved to {plot_path}')

    def calculate_num_mts_per_mtoc_over_time(self):
        """
        Reconstructs the number of pushing and pulling MTs attached to each
        MTOC at every accepted simulation step by replaying the spindle_trace.

        Returns:
            times (np.ndarray): time (s) of each step
            push_counts (dict[int, np.ndarray]): mtoc_id -> array of pushing MT counts at each step
            pull_counts (dict[int, np.ndarray]): mtoc_id -> array of pulling MT counts at each step
        """
        mtoc_ids = sorted(self.mtoc_positions.keys())
        push_state = np.zeros(self.num_push_sites)
        pull_state = np.zeros(self.num_pull_sites)

        times = []
        push_counts = {mtoc_id: [] for mtoc_id in mtoc_ids}
        pull_counts = {mtoc_id: [] for mtoc_id in mtoc_ids}

        for global_idx, (push, lattice_site, site_value) in self._replay_trace('spindle_trace'):
            if push:
                push_state[int(lattice_site)] = site_value
            else:
                pull_state[int(lattice_site)] = site_value
            times.append(global_idx * self.evolution_time)
            for mtoc_id in mtoc_ids:
                push_counts[mtoc_id].append(np.sum(push_state == mtoc_id))
                pull_counts[mtoc_id].append(np.sum(pull_state == mtoc_id))

        times = np.array(times)
        for mtoc_id in mtoc_ids:
            push_counts[mtoc_id] = np.array(push_counts[mtoc_id])
            pull_counts[mtoc_id] = np.array(pull_counts[mtoc_id])

        return times, push_counts, pull_counts

    def plot_num_mts_per_mtoc(self, save_path=None, stride=1, force='both', start_time=None, end_time=None, ylim=None):
        """
        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
            ylim (tuple): if given, (ymin, ymax) limits for the MT density axis (MTs / µm^2)

        Counts are divided by the boundary area 4 pi R^2, so the y-axis is MTs per µm^2.
        The returned push_counts / pull_counts are these densities.
        """
        if force == 'both':
            show_push = True
            show_pull = True
        elif force == 'push':
            show_push = True
            show_pull = False
        elif force == 'pull':
            show_push = False
            show_pull = True


        times, push_counts, pull_counts = self.calculate_num_mts_per_mtoc_over_time()

        mtoc_ids = sorted(push_counts.keys())

        # boundary is a sphere of radius boundary_radius
        area = 4 * np.pi * self.boundary_radius**2 # µm^2
        push_counts = {mtoc_id: push_counts[mtoc_id] / area for mtoc_id in mtoc_ids}
        pull_counts = {mtoc_id: pull_counts[mtoc_id] / area for mtoc_id in mtoc_ids}

        mask = np.ones_like(times, dtype=bool)
        if start_time is not None:
            mask &= times >= start_time
        if end_time is not None:
            mask &= times <= end_time
        times = times[mask]
        push_counts = {mtoc_id: push_counts[mtoc_id][mask] for mtoc_id in mtoc_ids}
        pull_counts = {mtoc_id: pull_counts[mtoc_id][mask] for mtoc_id in mtoc_ids}

        num_mtocs = len(mtoc_ids)
        shades = np.linspace(0.4, 0.9, num_mtocs) if num_mtocs > 1 else np.array([0.7])
        push_colors = plt.cm.Blues(shades)
        pull_colors = plt.cm.Reds(shades)

        fig, ax = plt.subplots(figsize=(6, 3))
        for i, mtoc_id in enumerate(mtoc_ids):
            if show_push:
                ax.plot(times[::stride], push_counts[mtoc_id][::stride], color=push_colors[i], alpha=1, label=f'MTOC {mtoc_id} pushing')
            if show_pull:
                ax.plot(times[::stride], pull_counts[mtoc_id][::stride], color=pull_colors[i], alpha=1, label=f'MTOC {mtoc_id} pulling')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'MTs attached per $\mu$m$^2$')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'num_mts_per_mtoc.png')
        plt.savefig(save_path)
        plt.close()
        print(f'MT density per MTOC plot saved to {save_path}')

        return times, push_counts, pull_counts

    def calculate_num_disk_mts_per_mtoc_over_time(self):
        """
        Reconstructs the number of pushing and pulling MTs attached to the
        metaphase plate (disk) for each MTOC at every accepted simulation
        step, by replaying the disk_trace (see optimize()). Front and back
        faces are combined into a single count per MTOC.

        Returns:
            times (np.ndarray): time (s) of each step
            push_counts (dict[int, np.ndarray]): mtoc_id -> array of disk pushing MT counts at each step
            pull_counts (dict[int, np.ndarray]): mtoc_id -> array of disk pulling MT counts at each step

        Raises:
            ValueError: if the plate is disabled, in which case no disk_trace was ever written.
        """
        if not self.disk_enabled:
            raise ValueError('this spindle was built with disk_enabled=False, so no disk MTs were ever attached')

        mtoc_ids = sorted(self.mtoc_positions.keys())
        push_state_front = np.zeros(self.num_disk_push_sites)
        push_state_back = np.zeros(self.num_disk_push_sites)
        pull_state_front = np.zeros(self.num_disk_pull_sites)
        pull_state_back = np.zeros(self.num_disk_pull_sites)

        times = []
        push_counts = {mtoc_id: [] for mtoc_id in mtoc_ids}
        pull_counts = {mtoc_id: [] for mtoc_id in mtoc_ids}

        for global_idx, (push, front, lattice_site, site_value) in self._replay_trace('disk_trace'):
            if push:
                state = push_state_front if front else push_state_back
            else:
                state = pull_state_front if front else pull_state_back
            state[int(lattice_site)] = site_value
            times.append(global_idx * self.evolution_time)
            for mtoc_id in mtoc_ids:
                push_counts[mtoc_id].append(np.sum(push_state_front == mtoc_id) + np.sum(push_state_back == mtoc_id))
                pull_counts[mtoc_id].append(np.sum(pull_state_front == mtoc_id) + np.sum(pull_state_back == mtoc_id))

        times = np.array(times)
        for mtoc_id in mtoc_ids:
            push_counts[mtoc_id] = np.array(push_counts[mtoc_id])
            pull_counts[mtoc_id] = np.array(pull_counts[mtoc_id])

        return times, push_counts, pull_counts

    def plot_num_disk_mts_per_mtoc(self, save_path=None, stride=1, force='both', start_time=None, end_time=None, ylim=None):
        """
        Same as plot_num_mts_per_mtoc, but for MTs attached to the metaphase
        plate (disk) instead of the boundary, combining front and back faces.

        Args:
            start_time (float): if given, restrict x-axis to times >= start_time
            end_time (float): if given, restrict x-axis to times <= end_time
            ylim (tuple): if given, (ymin, ymax) limits for the MT density axis (MTs / µm^2)

        Counts are divided by the total plate area 2 pi r^2 (both faces), so the y-axis is
        MTs per µm^2.

        Returns:
            (times, push_counts, pull_counts) as densities, or None when the plate is disabled.
        """
        if not self.disk_enabled:
            print('disk_enabled=False -- skipping plot_num_disk_mts_per_mtoc')
            return

        if force == 'both':
            show_push = True
            show_pull = True
        elif force == 'push':
            show_push = True
            show_pull = False
        elif force == 'pull':
            show_push = False
            show_pull = True

        times, push_counts, pull_counts = self.calculate_num_disk_mts_per_mtoc_over_time()

        mtoc_ids = sorted(push_counts.keys())

        # counts combine front and back faces, so normalize by both faces' area
        area = 2 * np.pi * self.disk_radius**2 # µm^2
        push_counts = {mtoc_id: push_counts[mtoc_id] / area for mtoc_id in mtoc_ids}
        pull_counts = {mtoc_id: pull_counts[mtoc_id] / area for mtoc_id in mtoc_ids}

        mask = np.ones_like(times, dtype=bool)
        if start_time is not None:
            mask &= times >= start_time
        if end_time is not None:
            mask &= times <= end_time
        times = times[mask]
        push_counts = {mtoc_id: push_counts[mtoc_id][mask] for mtoc_id in mtoc_ids}
        pull_counts = {mtoc_id: pull_counts[mtoc_id][mask] for mtoc_id in mtoc_ids}

        num_mtocs = len(mtoc_ids)
        shades = np.linspace(0.4, 0.9, num_mtocs) if num_mtocs > 1 else np.array([0.7])
        push_colors = plt.cm.Blues(shades)
        pull_colors = plt.cm.Reds(shades)

        fig, ax = plt.subplots(figsize=(6, 3))
        for i, mtoc_id in enumerate(mtoc_ids):
            if show_push:
                ax.plot(times[::stride], push_counts[mtoc_id][::stride], color=push_colors[i], alpha=1, label=f'MTOC {mtoc_id} pushing (disk)')
            if show_pull:
                ax.plot(times[::stride], pull_counts[mtoc_id][::stride], color=pull_colors[i], alpha=1, label=f'MTOC {mtoc_id} pulling (disk)')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'MTs attached per $\mu$m$^2$ (disk)')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'num_disk_mts_per_mtoc.png')
        plt.savefig(save_path)
        plt.close()
        print(f'disk MT density per MTOC plot saved to {save_path}')

        return times, push_counts, pull_counts

    def plot_pull_to_push_ratio(self, save_path=None, stride=1, per_mtoc=False, start_time=None, end_time=None):
        """
        Plots the ratio of pulling MTs to pushing MTs over time.

        Args:
            save_path (str): where to save the plot. Defaults to plot_folder_path/pull_to_push_ratio.png
            stride (int): subsample factor for plotting
            per_mtoc (bool): if True, plot a separate ratio curve per MTOC in addition to the total ratio
            start_time (float): if given, restrict to times >= start_time
            end_time (float): if given, restrict to times <= end_time
        """
        times, push_counts, pull_counts = self.calculate_num_mts_per_mtoc_over_time()
        mtoc_ids = sorted(push_counts.keys())

        mask = np.ones_like(times, dtype=bool)
        if start_time is not None:
            mask &= times >= start_time
        if end_time is not None:
            mask &= times <= end_time
        times = times[mask]
        push_counts = {mtoc_id: push_counts[mtoc_id][mask] for mtoc_id in mtoc_ids}
        pull_counts = {mtoc_id: pull_counts[mtoc_id][mask] for mtoc_id in mtoc_ids}

        total_push = np.sum([push_counts[mtoc_id] for mtoc_id in mtoc_ids], axis=0)
        total_pull = np.sum([pull_counts[mtoc_id] for mtoc_id in mtoc_ids], axis=0)
        total_ratio = np.divide(total_pull, total_push, out=np.full_like(total_pull, np.nan, dtype=float), where=total_push != 0)

        mean_ratio = np.nanmean(total_ratio)
        std_ratio = np.nanstd(total_ratio)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(times[::stride], total_ratio[::stride], color='black', label='total')
        ax.axhline(mean_ratio, color='black', linestyle='--', linewidth=1, label=f'mean = {mean_ratio:.2f}')
        ax.fill_between(times[::stride], mean_ratio - std_ratio, mean_ratio + std_ratio, color='black', alpha=0.15, label=f'std = {std_ratio:.2f}')

        if per_mtoc:
            num_mtocs = len(mtoc_ids)
            shades = np.linspace(0.4, 0.9, num_mtocs) if num_mtocs > 1 else np.array([0.7])
            colors = plt.cm.Purples(shades)
            for i, mtoc_id in enumerate(mtoc_ids):
                ratio = np.divide(pull_counts[mtoc_id], push_counts[mtoc_id], out=np.full_like(pull_counts[mtoc_id], np.nan, dtype=float), where=push_counts[mtoc_id] != 0)
                ax.plot(times[::stride], ratio[::stride], color=colors[i], alpha=0.8, label=f'MTOC {mtoc_id}')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('# pulling MTs / # pushing MTs')
        ax.legend()
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'pull_to_push_ratio.png')
        plt.savefig(save_path)
        plt.close()
        print(f'pull/push ratio plot saved to {save_path}')

        return times, total_ratio, mean_ratio, std_ratio

    def _trace_batches(self, prefix):
        """
        The batch files of one trace log, in simulation order, as
        (start, end, changes): start/end are the global accepted-state indices
        the batch spans -- flush_accepted_states() encodes them in the filename
        -- and changes is the loaded array of rows.
        """
        spindle_trace_dir = os.path.join(self.dir_path, 'spindle_trace')

        def batch_range(path):
            start, end = os.path.splitext(os.path.basename(path))[0].split('_')[-2:]
            return int(start), int(end)

        paths = sorted(glob.glob(os.path.join(spindle_trace_dir, f'{prefix}_*.npy')), key=batch_range)
        for path in paths:
            start, end = batch_range(path)
            yield start, end, np.load(path, allow_pickle=True)

    def _replay_trace(self, prefix):
        """
        Yields (global_idx, change) for every row of one trace log, where
        global_idx is the accepted-state index (i.e. time / evolution_time) the
        change belongs to.

        Every accepted step advances the clock by evolution_time but appends
        its change to exactly one of the two logs -- spindle_trace for boundary
        sites, disk_trace for metaphase-plate sites (see optimize()). A row's
        position within its own log is therefore *not* the global step index:
        counting rows makes the reconstructed time axis lag the simulation by
        the fraction of steps that went to the other log, so late-time windows
        run off the end of the log and read as empty. Each batch filename
        records the global range it covers, so anchor on those and spread a
        batch's rows evenly across its range (how the two logs interleave
        within a batch isn't recorded).
        """
        for start, end, changes in self._trace_batches(prefix):
            num_rows = len(changes)
            if num_rows == 0:
                continue
            span = max(end - start, num_rows)
            for i, change in enumerate(changes):
                yield start + (i * span) // num_rows, change

    def _calculate_occupancy(self, mtoc_id, start_time, end_time, push):
        """
        Fraction of the accepted steps in [start_time, end_time] during which
        each site of the push (push=True) or pull (push=False) boundary lattice
        was held by mtoc_id.
        """
        start_idx = int(np.round(start_time, 4) / self.evolution_time)
        end_idx = int(np.round(end_time, 4) / self.evolution_time)

        num_sites = self.num_push_sites if push else self.num_pull_sites
        state = np.zeros(num_sites)
        occupancy = np.zeros(num_sites)

        # a state persists from its own change until the next one, which is
        # generally more than a single step: the steps in between updated the
        # disk instead, and their rows live in the other log. So weight each
        # state by the number of global steps it actually held for, which is
        # what normalizing by (end_idx - start_idx) below assumes.
        prev_idx = 0
        for global_idx, (is_push, lattice_site, site_value) in self._replay_trace('spindle_trace'):
            dwell = min(global_idx, end_idx) - max(prev_idx, start_idx)
            if dwell > 0:
                occupancy += dwell * (state == mtoc_id)
            if global_idx >= end_idx:
                break
            if is_push == push:
                state[int(lattice_site)] = site_value
            prev_idx = global_idx
        else:
            # the log ended before end_idx (a trailing batch that never filled,
            # see optimize()) -- its final state holds for the rest of the window
            dwell = end_idx - max(prev_idx, start_idx)
            if dwell > 0:
                occupancy += dwell * (state == mtoc_id)

        num_states = end_idx - start_idx
        if num_states > 0:
            occupancy = occupancy / num_states
        return occupancy

    def calculate_motor_occupancy(self, mtoc_id, start_time, end_time):
        return self._calculate_occupancy(mtoc_id, start_time, end_time, push=False)

    def calculate_push_occupancy(self, mtoc_id, start_time, end_time):
        return self._calculate_occupancy(mtoc_id, start_time, end_time, push=True)

    def _calculate_lifetime_samples(self, mtoc_id, start_time, end_time, push):
        """
        Replays the spindle_trace and records every individual MT death event
        for mtoc_id within [start_time, end_time], on either the push or pull
        lattice.

        Args:
            push (bool): True to sample the push lattice, False for pull.

        Returns:
            site_indices (np.ndarray[int]): lattice site of each death event.
            lifetimes (np.ndarray[float]): lifetime (s) of each death event,
            same order/length as site_indices.
        """
        start_idx = int(np.round(start_time, 4) / self.evolution_time)
        end_idx = int(np.round(end_time, 4) / self.evolution_time)

        num_sites = self.num_push_sites if push else self.num_pull_sites
        state = np.zeros(num_sites)
        birth_time = np.full(num_sites, np.nan)
        site_indices = []
        lifetimes = []

        for global_idx, (is_push, lattice_site, site_value) in self._replay_trace('spindle_trace'):
            if global_idx >= end_idx:
                break
            if is_push == push:
                lattice_site = int(lattice_site)
                current_time = global_idx * self.evolution_time
                old_value = state[lattice_site]
                if site_value != 0 and old_value == 0:
                    birth_time[lattice_site] = current_time
                elif site_value == 0 and old_value == mtoc_id and global_idx >= start_idx:
                    site_indices.append(lattice_site)
                    lifetimes.append(current_time - birth_time[lattice_site])
                state[lattice_site] = site_value

        return np.array(site_indices, dtype=int), np.array(lifetimes)

    def calculate_push_lifetime(self, mtoc_id, start_time, end_time):
        """
        Mean lifetime (s) of pushing MTs from mtoc_id at each push lattice
        site, over MTs that died (were removed from that site) within
        [start_time, end_time].

        Returns:
            np.ndarray: shape (num_push_sites,); 0 at sites with no MT deaths
            recorded in the window.
        """
        site_indices, lifetimes = self._calculate_lifetime_samples(mtoc_id, start_time, end_time, push=True)
        mean_lifetime = np.zeros(self.num_push_sites)
        if lifetimes.size:
            sums = np.bincount(site_indices, weights=lifetimes, minlength=self.num_push_sites)
            counts = np.bincount(site_indices, minlength=self.num_push_sites)
            has_data = counts > 0
            mean_lifetime[has_data] = sums[has_data] / counts[has_data]
        return mean_lifetime

    def calculate_motor_lifetime(self, mtoc_id, start_time, end_time):
        """
        Mean lifetime (s) of pulling MTs from mtoc_id at each pull lattice
        site, over MTs that died (were removed from that site) within
        [start_time, end_time].

        Returns:
            np.ndarray: shape (num_pull_sites,); 0 at sites with no MT deaths
            recorded in the window.
        """
        site_indices, lifetimes = self._calculate_lifetime_samples(mtoc_id, start_time, end_time, push=False)
        mean_lifetime = np.zeros(self.num_pull_sites)
        if lifetimes.size:
            sums = np.bincount(site_indices, weights=lifetimes, minlength=self.num_pull_sites)
            counts = np.bincount(site_indices, minlength=self.num_pull_sites)
            has_data = counts > 0
            mean_lifetime[has_data] = sums[has_data] / counts[has_data]
        return mean_lifetime

    def plot_lifetime_surface(self, mtoc_id, start_time, end_time, show_occupancy='pull'):
        """
        Spatial map of mean MT lifetime (s) at each lattice site, for MTs
        from mtoc_id that died within [start_time, end_time]. Same
        layout/conventions as plot_surface_occupancy, but colored by
        lifetime instead of occupancy proportion.

        show_occupancy: one of 'pull', 'push', 'both' (True is treated as 'both').
        Controls which lattices are plotted.
        """
        if show_occupancy is True:
            show_occupancy = 'both'
        show_occupancy = show_occupancy.lower()
        if show_occupancy not in ('pull', 'push', 'both'):
            raise ValueError("show_occupancy must be one of 'pull', 'push', 'both'/True")
        show_pull = show_occupancy in ('pull', 'both')
        show_push = show_occupancy in ('push', 'both')

        fig = plt.figure()
        # computed_zorder=False makes draw order (not true 3D depth) determine what's on top,
        # so the pull scatter (drawn last) stays visible over the far denser push scatter.
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        plt.title(f'MT lifetime (s) for MTOC {mtoc_id}\nfrom {start_time} to {end_time} seconds')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('z (um)')

        if show_push:
            push_lifetime = self.calculate_push_lifetime(mtoc_id, start_time, end_time)
            push_lattice = self.push_lattice
            push_norm = mcolors.Normalize(vmin=push_lifetime.min(), vmax=push_lifetime.max())
            push_cmap = plt.cm.viridis
            push_rgba = push_cmap(push_norm(push_lifetime))
            # scale alpha with lifetime itself so sites with no recorded death fade toward invisible
            push_rgba[:, 3] = push_norm(push_lifetime) * 0.5
            ax.scatter(push_lattice[:,0], push_lattice[:,1], push_lattice[:,2],
                       c=push_rgba, zorder=1)
            mappable = plt.cm.ScalarMappable(cmap=push_cmap, norm=push_norm)
            mappable.set_array(push_lifetime)
            fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.15, label="Push MT lifetime (s)")

        if show_pull:
            pull_lifetime = self.calculate_motor_lifetime(mtoc_id, start_time, end_time)
            pull_lattice = self.pull_lattice
            pull_norm = mcolors.Normalize(vmin=pull_lifetime.min(), vmax=pull_lifetime.max())
            pull_cmap = plt.cm.inferno
            ax.scatter(pull_lattice[:,0], pull_lattice[:,1], pull_lattice[:,2],
                       c=pull_lifetime, alpha=0.7, cmap=pull_cmap, norm=pull_norm, zorder=2)
            mappable = plt.cm.ScalarMappable(cmap=pull_cmap, norm=pull_norm)
            mappable.set_array(pull_lifetime)
            fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1, label="Pull MT lifetime (s)")

        ax.axis('equal')

        plot_path = os.path.join(self.plot_folder_path, f'lifetime_surface_{show_occupancy}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'MT lifetime surface plot saved to {plot_path}')

    def plot_lifetime_vs_angle(self, mtoc_id, start_time, end_time, show_occupancy='pull', num_bins=32):
        """
        Plots mean MT lifetime (s), binned by polar angle (radians) from the
        axis running through the sphere center and the surface point closest
        to the given MTOC (i.e. the axis along the MTOC's own position
        vector). Same layout/conventions as plot_occupancy_vs_angle, but
        showing lifetime instead of occupancy proportion.

        show_occupancy: one of 'pull', 'push', 'both' (True is treated as 'both').
        """
        if show_occupancy is True:
            show_occupancy = 'both'
        show_occupancy = show_occupancy.lower()
        if show_occupancy not in ('pull', 'push', 'both'):
            raise ValueError("show_occupancy must be one of 'pull', 'push', 'both'/True")
        show_pull = show_occupancy in ('pull', 'both')
        show_push = show_occupancy in ('push', 'both')

        axis, _ = normalize_vecs(self.mtoc_positions[mtoc_id])

        def _angles_rad(lattice):
            unit_lattice, _ = normalize_vecs(lattice)
            cos_angle = np.clip(unit_lattice @ axis, -1.0, 1.0)
            return np.arccos(cos_angle)

        bin_edges = np.linspace(0, np.pi, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = np.pi / num_bins

        def _binned_mean_std(angles, lifetimes):
            # bins over individual MT death events (not per-site means), so
            # the error bars reflect the spread of actual MT lifetimes
            bin_idx = np.clip(np.digitize(angles, bin_edges) - 1, 0, num_bins - 1)
            means = np.zeros(num_bins)
            stds = np.zeros(num_bins)
            for i in range(num_bins):
                mask = bin_idx == i
                if np.any(mask):
                    means[i] = lifetimes[mask].mean()
                    stds[i] = lifetimes[mask].std()
            return means, stds

        fig, ax = plt.subplots()
        plt.title(f'MT lifetime vs angle from MTOC {mtoc_id} axis, {start_time} to {end_time} seconds')

        if show_pull:
            pull_sites, pull_lifetimes = self._calculate_lifetime_samples(mtoc_id, start_time, end_time, push=False)
            pull_angles = _angles_rad(self.pull_lattice[pull_sites])
            pull_mean, pull_std = _binned_mean_std(pull_angles, pull_lifetimes)
            ax.bar(bin_centers, pull_mean, width=bin_width, alpha=0.6,
                   color='tab:red', edgecolor='tab:red', label='Pull MT lifetime',
                   yerr=pull_std, capsize=3, ecolor='tab:red')

        if show_push:
            push_sites, push_lifetimes = self._calculate_lifetime_samples(mtoc_id, start_time, end_time, push=True)
            push_angles = _angles_rad(self.push_lattice[push_sites])
            push_mean, push_std = _binned_mean_std(push_angles, push_lifetimes)
            ax.bar(bin_centers, push_mean, width=bin_width, alpha=0.6,
                   color='tab:blue', edgecolor='tab:blue', label='Push MT lifetime',
                   yerr=push_std, capsize=3, ecolor='tab:blue')

        ax.set_xlim(0, np.pi)
        ax.set_xlabel('angle from MTOC axis (radians)')
        ax.set_ylabel('mean MT lifetime (s)')
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(self.plot_folder_path, f'lifetime_vs_angle.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'MT lifetime vs angle plot saved to {plot_path}')

    def animate_mtoc_trajectory(self, save_path=None, interval=50, stride=100, show_occupancy=False, occupancy_mtoc_id=1, start_time=None, end_time=None):
        """
        show_occupancy: one of False/None/'none', 'pull', 'push', 'both' (True is treated as 'both').
        Controls which motor occupancy overlays are drawn.

        start_time/end_time: if given, restrict the animation to trajectory
        entries with start_time <= t <= end_time (in simulated/lab seconds)
        instead of the full recorded trajectory.
        """
        if show_occupancy is True:
            show_occupancy = 'both'
        elif not show_occupancy:
            show_occupancy = 'none'
        show_occupancy = show_occupancy.lower()
        if show_occupancy not in ('none', 'pull', 'push', 'both'):
            raise ValueError("show_occupancy must be one of False, None, 'none', 'pull', 'push', 'both'/True")
        show_pull = show_occupancy in ('pull', 'both')
        show_push = show_occupancy in ('push', 'both')

        print(list(self.trajectory.keys())[-1])
        times = sorted(self.trajectory.keys())
        # drop the most recent 1000 raw trajectory keys (their trailing tail can still be in
        # flux) -- but only if there are enough entries to spare, so short trajectories aren't
        # emptied out entirely
        if len(times) > 1000:
            times = times[:-1000]
        if start_time is not None:
            times = [t for t in times if t >= start_time]
        if end_time is not None:
            times = [t for t in times if t <= end_time]
        if not times:
            raise ValueError(f"No trajectory entries found in range [{start_time}, {end_time}]")
        times = times[::stride]
        mtoc_ids = sorted(self.trajectory[times[0]]['mtoc_pos'].keys())
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax = fig.add_subplot(gs[0], projection='3d')
        ax_info = fig.add_subplot(gs[1])
        ax_info.axis('off')

        # wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        xs = self.boundary_radius * np.outer(np.cos(u), np.sin(v))
        ys = self.boundary_radius * np.outer(np.sin(u), np.sin(v))
        zs = self.boundary_radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.2, linewidth=0.5)

        # -- metaphase plate (disk): pose (+ characteristic vectors) and site occupancy --
        # all of this is skipped when the plate is disabled; disk_artists stays empty and
        # update() leaves it alone, so the animation is just the MTOCs and the boundary
        disk_artists = []
        has_disk_pose_history = self.disk_enabled and 'disk_center' in self.trajectory[times[0]]
        if self.disk_enabled and not has_disk_pose_history:
            print("No per-frame disk pose was recorded in this trajectory (predates disk "
                  "tracking) -- showing the disk at its current, static pose in every frame.")

        def _disk_pose(t):
            if has_disk_pose_history:
                data = self.trajectory[t]
                return data['disk_center'], data['disk_e1'], data['disk_e2'], data['disk_e3']
            return self.disk_center, self.disk_e1, self.disk_e2, self.disk_e3

        def _disk_ring(center, e2, e3, n=48):
            theta = np.linspace(0, 2 * np.pi, n)
            return center + self.disk_radius * (np.outer(np.cos(theta), e2) + np.outer(np.sin(theta), e3))

        def _disk_axis_ends(center, e1, e2, e3):
            length = self.disk_radius
            return (np.array([center, center + length * e1]),
                    np.array([center, center + length * e2]),
                    np.array([center, center + length * e3]))

        if self.disk_enabled:
            center0, e1_0, e2_0, e3_0 = _disk_pose(times[0])

            disk_patch = Poly3DCollection([_disk_ring(center0, e2_0, e3_0)],
                                           facecolor='tab:blue', alpha=0.25, edgecolor='navy')
            ax.add_collection3d(disk_patch)

            e1_ends0, e2_ends0, e3_ends0 = _disk_axis_ends(center0, e1_0, e2_0, e3_0)
            e1_line, = ax.plot(*e1_ends0.T, color='navy', linewidth=2, label='disk e1 (normal)')
            e2_line, = ax.plot(*e2_ends0.T, color='indigo', linewidth=1.5, label='disk e2')
            e3_line, = ax.plot(*e3_ends0.T, color='mediumvioletred', linewidth=1.5, label='disk e3')

            # every disk site (both faces, push + pull), so the tessellation and its occupancy are
            # both visible: empty sites are drawn small/faint, occupied ones large/opaque and
            # colored to match the mtoc occupying them. Occupancy itself is static across frames
            # for now (the disk isn't wired into the sampler yet, see sample_spindle_update) -- only
            # each site's lab-frame position needs to be recomputed per frame as the disk moves.
            disk_site_local = np.concatenate([self.disk_push_local, self.disk_push_local,
                                               self.disk_pull_local, self.disk_pull_local], axis=0)
            disk_site_state = np.concatenate([self.disk_push_state_front, self.disk_push_state_back,
                                               self.disk_pull_state_front, self.disk_pull_state_back])

            def _disk_site_positions(center, e2, e3):
                a, b = disk_site_local[:, 0], disk_site_local[:, 1]
                return center + a[:, np.newaxis] * e2 + b[:, np.newaxis] * e3

            disk_site_colors = np.tile(mcolors.to_rgba('lightgray', alpha=0.15), (len(disk_site_state), 1))
            disk_site_sizes = np.full(len(disk_site_state), 4.0)
            for i, mtoc_id in enumerate(mtoc_ids):
                occupied = disk_site_state == mtoc_id
                disk_site_colors[occupied] = mcolors.to_rgba(colors[i % len(colors)], alpha=0.9)
                disk_site_sizes[occupied] = 25.0

            disk_sites_scatter = ax.scatter(*_disk_site_positions(center0, e2_0, e3_0).T,
                                             c=disk_site_colors, s=disk_site_sizes)

            disk_artists = [disk_patch, e1_line, e2_line, e3_line, disk_sites_scatter]

            def _update_disk(t):
                center, e1, e2, e3 = _disk_pose(t)
                disk_patch.set_verts([_disk_ring(center, e2, e3)])
                e1_ends, e2_ends, e3_ends = _disk_axis_ends(center, e1, e2, e3)
                e1_line.set_data_3d(*e1_ends.T)
                e2_line.set_data_3d(*e2_ends.T)
                e3_line.set_data_3d(*e3_ends.T)
                disk_sites_scatter._offsets3d = tuple(_disk_site_positions(center, e2, e3).T)
        else:
            def _update_disk(t):
                pass

        # precompute per-frame occupancy
        occupancy_frames = None
        push_occupancy_frames = None
        occ_scatter = None
        push_occ_scatter = None
        if show_pull or show_push:
            window = stride * self.evolution_time
            last_time = sorted(self.trajectory.keys())[-1]
            print(f'Precomputing occupancy for {len(times)} frames...')
            occupancy_frames = [] if show_pull else None
            push_occupancy_frames = [] if show_push else None
            for i, t in tqdm.tqdm(enumerate(times)):
                t_end = t + window
                if t_end > last_time:
                    # the window runs past the end of the recording: hold the
                    # previous frame, or clamp if this is the very first one
                    if i > 0:
                        if show_pull:
                            occupancy_frames.append(occupancy_frames[-1])
                        if show_push:
                            push_occupancy_frames.append(push_occupancy_frames[-1])
                        continue
                    t_end = last_time
                if show_pull:
                    occupancy_frames.append(self.calculate_motor_occupancy(occupancy_mtoc_id, t, t_end))
                if show_push:
                    push_occupancy_frames.append(self.calculate_push_occupancy(occupancy_mtoc_id, t, t_end))
            if show_pull:
                occupancy_frames = np.array(occupancy_frames)
            if show_push:
                push_occupancy_frames = np.array(push_occupancy_frames)
            print('Done.')

            print('Rendering Animation . . . ')

            if show_push:
                push_cmap = plt.cm.Blues
                push_occ_norm = mcolors.Normalize(vmin=0, vmax=max(push_occupancy_frames.max(), 1e-9))
                px, py, pz = self.push_lattice[:, 0], self.push_lattice[:, 1], self.push_lattice[:, 2]

                def _push_rgba(occ):
                    rgba = push_cmap(push_occ_norm(occ))
                    rgba[occ == 0, 3] = 0    # transparent where never occupied
                    rgba[occ > 0, 3] = 0.2   # low alpha for occupied push sites
                    return rgba

                push_occ_scatter = ax.scatter(px, py, pz, c=_push_rgba(push_occupancy_frames[0]), s=5)
                fig.colorbar(plt.cm.ScalarMappable(norm=push_occ_norm, cmap=push_cmap), ax=ax, shrink=0.4, pad=0.10, label=f'Push occupancy (MTOC {occupancy_mtoc_id})')

            if show_pull:
                pull_cmap = plt.cm.Reds
                occ_norm = mcolors.Normalize(vmin=0, vmax=max(occupancy_frames.max(), 1e-9))
                lx, ly, lz = self.pull_lattice[:, 0], self.pull_lattice[:, 1], self.pull_lattice[:, 2]

                def _pull_rgba(occ):
                    rgba = pull_cmap(occ_norm(occ))
                    rgba[occ == 0, 3] = 0    # transparent where never occupied
                    rgba[occ > 0, 3] = 0.7   # higher alpha so pull sites read clearly over push
                    return rgba

                occ_scatter = ax.scatter(lx, ly, lz, c=_pull_rgba(occupancy_frames[0]), s=10)
                fig.colorbar(plt.cm.ScalarMappable(norm=occ_norm, cmap=pull_cmap), ax=ax, shrink=0.4, pad=0.15, label=f'Pull occupancy (MTOC {occupancy_mtoc_id})')

        scatters = []
        for i, mtoc_id in enumerate(mtoc_ids):
            pos = self.trajectory[times[0]]['mtoc_pos'][mtoc_id]
            sc = ax.scatter(*pos, s=60, color=colors[i % len(colors)], label=f'MTOC {mtoc_id}')
            scatters.append(sc)

        r = self.boundary_radius
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_zlabel('z (µm)')
        ax.axis('equal')
        handles, labels = ax.get_legend_handles_labels()
        ax_info.legend(handles, labels, loc='upper right', fontsize=8)

        def _stats(t):
            data = self.trajectory[t]
            mtoc_pos = data['mtoc_pos']
            cost = data['cost']
            tubulin_use = data.get('tubulin_use', float('nan'))
            num_mts = data.get('num_mts', float('nan'))
            if len(mtoc_ids) >= 2:
                diff = mtoc_pos[mtoc_ids[0]] - mtoc_pos[mtoc_ids[1]]
                dist = float(np.linalg.norm(diff))
            else:
                dist = float('nan')
            forces = data.get('total_force', {})
            return cost, tubulin_use, num_mts, dist, forces

        def _format_text(t, cost, tubulin_use, num_mts, dist, forces):
            lines = (
                f't              = {t:.4f} s\n'
                f'cost           = {cost:.4f}\n'
                f'tubulin use    = {tubulin_use:.2f} / {self.tubulin_budget:.2f} µm\n'
                f'# MTs          = {num_mts}\n'
                f'separation     = {dist:.3f} µm'
            )
            for mid in mtoc_ids:
                f = forces.get(mid)
                mag = float(np.linalg.norm(f)) if f is not None else float('nan')
                lines += f'\n|F| MTOC {mid}     = {mag:.2f} pN'
            return lines

        cost0, tub0, nmts0, dist0, forces0 = _stats(times[0])
        info_text = ax_info.text(
            -0.4, 0.7,
            _format_text(times[0], cost0, tub0, nmts0, dist0, forces0),
            transform=ax_info.transAxes,
            verticalalignment='top',
            fontsize=10,
            fontfamily='monospace',
        )

        def update(frame):
            t = times[frame]
            mtoc_pos = self.trajectory[t]['mtoc_pos']
            for sc, mtoc_id in zip(scatters, mtoc_ids):
                pos = mtoc_pos[mtoc_id]
                sc._offsets3d = (np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]))

            _update_disk(t)

            if occ_scatter is not None:
                occ_scatter.set_facecolors(_pull_rgba(occupancy_frames[frame]))
            if push_occ_scatter is not None:
                push_occ_scatter.set_facecolors(_push_rgba(push_occupancy_frames[frame]))
            cost, tubulin_use, num_mts, dist, forces = _stats(t)
            info_text.set_text(_format_text(t, cost, tubulin_use, num_mts, dist, forces))
            return (scatters + disk_artists
                    + ([occ_scatter] if occ_scatter is not None else [])
                    + ([push_occ_scatter] if push_occ_scatter is not None else []) + [info_text])

        anim = FuncAnimation(fig, update, frames=len(times), interval=interval, blit=False)

        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=1000 // interval)
            print(f'animation saved to {save_path}')
            plt.close()
        else:
            plt.show()

        return anim

    def plot_mtoc_trajectory_frame(self, t, window, save_path=None, show_occupancy=False, occupancy_mtoc_id=1):
        """
        Plots a single frame of animate_mtoc_trajectory: MTOC positions at the
        trajectory entry closest to time t, optionally overlaid with motor
        occupancy accumulated over the window [t - window, t].

        show_occupancy: one of False/None/'none', 'pull', 'push', 'both' (True is treated as 'both').
        """
        if show_occupancy is True:
            show_occupancy = 'both'
        elif not show_occupancy:
            show_occupancy = 'none'
        show_occupancy = show_occupancy.lower()
        if show_occupancy not in ('none', 'pull', 'push', 'both'):
            raise ValueError("show_occupancy must be one of False, None, 'none', 'pull', 'push', 'both'/True")
        show_pull = show_occupancy in ('pull', 'both')
        show_push = show_occupancy in ('push', 'both')

        times = sorted(self.trajectory.keys())
        t = min(times, key=lambda candidate: abs(candidate - t))
        window_start = max(t - window, times[0])
        mtoc_ids = sorted(self.trajectory[t]['mtoc_pos'].keys())
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax = fig.add_subplot(gs[0], projection='3d', computed_zorder=False)
        ax_info = fig.add_subplot(gs[1])
        ax_info.axis('off')

        # wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        xs = self.boundary_radius * np.outer(np.cos(u), np.sin(v))
        ys = self.boundary_radius * np.outer(np.sin(u), np.sin(v))
        zs = self.boundary_radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.2, linewidth=0.5, zorder=0)

        if show_push:
            push_occupancy = self.calculate_push_occupancy(occupancy_mtoc_id, window_start, t)
            push_cmap = plt.cm.Blues
            push_norm = mcolors.Normalize(vmin=0, vmax=max(push_occupancy.max(), 1e-9))
            px, py, pz = self.push_lattice[:, 0], self.push_lattice[:, 1], self.push_lattice[:, 2]
            push_rgba = push_cmap(push_norm(push_occupancy))
            push_rgba[push_occupancy == 0, 3] = 0    # transparent where never occupied
            push_rgba[push_occupancy > 0, 3] = 0.2   # low alpha for occupied push sites
            ax.scatter(px, py, pz, c=push_rgba, s=5, zorder=1)
            fig.colorbar(plt.cm.ScalarMappable(norm=push_norm, cmap=push_cmap), ax=ax, shrink=0.4, pad=0.10, label=f'Push occupancy (MTOC {occupancy_mtoc_id})')

        if show_pull:
            pull_occupancy = self.calculate_motor_occupancy(occupancy_mtoc_id, window_start, t)
            pull_cmap = plt.cm.Reds
            pull_norm = mcolors.Normalize(vmin=0, vmax=max(pull_occupancy.max(), 1e-9))
            lx, ly, lz = self.pull_lattice[:, 0], self.pull_lattice[:, 1], self.pull_lattice[:, 2]
            pull_rgba = pull_cmap(pull_norm(pull_occupancy))
            pull_rgba[pull_occupancy == 0, 3] = 0    # transparent where never occupied
            pull_rgba[pull_occupancy > 0, 3] = 0.7   # higher alpha so pull sites read clearly over push
            ax.scatter(lx, ly, lz, c=pull_rgba, s=10, zorder=2)
            fig.colorbar(plt.cm.ScalarMappable(norm=pull_norm, cmap=pull_cmap), ax=ax, shrink=0.4, pad=0.15, label=f'Pull occupancy (MTOC {occupancy_mtoc_id})')

        for i, mtoc_id in enumerate(mtoc_ids):
            pos = self.trajectory[t]['mtoc_pos'][mtoc_id]
            ax.scatter(*pos, s=60, color=colors[i % len(colors)], label=f'MTOC {mtoc_id}', zorder=3)

        r = self.boundary_radius
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_zlabel('z (µm)')
        ax.axis('equal')
        ax.legend()
        ax.set_title(f't={np.round(t)}\n occupancy for t=[{np.round(t-window)}, {np.round(t)}]')

        data = self.trajectory[t]
        mtoc_pos = data['mtoc_pos']
        cost = data['cost']
        tubulin_use = data.get('tubulin_use', float('nan'))
        num_mts = data.get('num_mts', float('nan'))
        if len(mtoc_ids) >= 2:
            diff = mtoc_pos[mtoc_ids[0]] - mtoc_pos[mtoc_ids[1]]
            dist = float(np.linalg.norm(diff))
        else:
            dist = float('nan')
        forces = data.get('total_force', {})

        lines = (
            # f't              = {t:.4f} s\n'
            # f'cost           = {cost:.4f}\n'
            # f'tubulin use    = {tubulin_use:.2f} / {self.tubulin_budget:.2f} µm\n'
            # f'# MTs          = {num_mts}\n'
            # f'separation     = {dist:.3f} µm'
            ''
        )
        # if show_pull or show_push:
        #     lines += f'\noccupancy window = [{window_start:.4f}, {t:.4f}] s'
        # for mid in mtoc_ids:
        #     f = forces.get(mid)
        #     mag = float(np.linalg.norm(f)) if f is not None else float('nan')
        #     lines += f'\n|F| MTOC {mid}     = {mag:.2f} pN'

        ax_info.text(
            -0.4, 0.7,
            lines,
            transform=ax_info.transAxes,
            verticalalignment='top',
            fontsize=10,
            fontfamily='monospace',
        )

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, f'mtoc_trajectory_frame_t{t:.4f}_{show_occupancy}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f'mtoc trajectory frame saved to {save_path}')

        return fig