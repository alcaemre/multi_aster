#
# Emre Alca
# University of Pennsylvania
# Created on Wed Jun 03 2026
# Last Modified: 2026/06/30 16:07:40
#

import numpy as np
np.set_printoptions(formatter={'float': '{:.3f}'.format})
# import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import time
from datetime import datetime

import pickle
import os
import glob
import psutil
from scipy.spatial import cKDTree

from rich.console import Console
from rich.live import Live
from rich.table import Table
from matplotlib.animation import FuncAnimation

console = Console()


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
    

def retrieve_experiement(experiment_dir, save_trajectory=False, save=False, max_states=None):

    num_attempts_path = os.path.join(experiment_dir, 'num_attempts.npy')
    num_attempts = np.load(num_attempts_path)

    num_accepted_states_path = os.path.join(experiment_dir, 'num_accepted_states.npy')
    num_accepted_states = np.load(num_accepted_states_path)

    # -- reinitialize spindle from saved spindle dict and trajectory --
    spindle_path = os.path.join(experiment_dir, 'spindle.pkl')
    with open(spindle_path, 'rb') as f:
        spindle_dict = pickle.load(f)

    trajectory_path = os.path.join(experiment_dir, 'trajectory.pkl')
    with open(trajectory_path, 'rb') as f:
        trajectory = pickle.load(f)

    last_time = list(trajectory.keys())[-1]
    last_mtoc_positions = np.array(list(trajectory[last_time]['mtoc_pos'].values()))

    num_accepted_states = num_accepted_states // 1000 * 1000

    push_batch_name = trace_batch_name('push', num_accepted_states - 1000, num_accepted_states)
    push_batch_path = os.path.join(experiment_dir, 'spindle_trace')
    push_batch_path = os.path.join(push_batch_path, push_batch_name)
    last_push_state = np.load(push_batch_path)[-1]

    pull_batch_name = trace_batch_name('pull', num_accepted_states - 1000, num_accepted_states)
    pull_batch_path = os.path.join(experiment_dir, pull_batch_name)
    pull_batch_path = os.path.join(experiment_dir, 'spindle_trace')
    pull_batch_path = os.path.join(pull_batch_path, pull_batch_name)
    last_pull_state = np.load(pull_batch_path)[-1]

    spindle_from_dict = Spindle(
        initial_mtoc_positions=last_mtoc_positions,
        push_lattice=spindle_dict['push_lattice'],
        pull_lattice=spindle_dict['pull_lattice'],
        initial_time=last_time, # s
    
        # -- optimization paramters --
        tubulin_budget=spindle_dict['tubulin_budget'], # µm
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
    
        # -- saving info --
        seed=spindle_dict['seed'],  # random seed
        dir_path=experiment_dir, # directory of existing experiment
        data_dir = spindle_dict['path_to_data'], # directory within which each experiment's data is stored
        save_trajectory=save_trajectory, # True if you want the system to save the whole trajectory. Used to make animations and plots
        save=save,
        trajectory=trajectory,
        )
    
    spindle_from_dict.push_state = last_push_state
    spindle_from_dict.pull_state = last_pull_state
    
    # with Live(console=console, refresh_per_second=4) as live:
    #     outer_table = Table(title="Spindle Simulation")
    #     outer_table.add_column("Parameter", justify="left")
    #     outer_table.add_column("Value", justify="right")
    #     outer_table.add_column("Parameter", justify="left")
    #     outer_table.add_column("Value", justify="right")
        
    #     # set table values
    #     outer_table.add_row('Last Accepted Time (s)', str(spindle_from_dict.time)) # last stable time
    #     outer_table.add_row('Last Accepted Position (um)', str(spindle_from_dict.mtoc_positions)) # last stable position
    #     outer_table.add_row('distance between asters', f'{normalize_vecs(spindle_from_dict.mtoc_positions[1] - spindle_from_dict.mtoc_positions[2])[1]}') # tubulin use / tubulin budget
    #     outer_table.add_row('Last Accepted Cost', str(spindle_from_dict.cost)) # last accepted cost
    #     outer_table.add_row('Number of MTs', str(spindle_from_dict.calculate_num_mts())) # number of MTs
    #     outer_table.add_row('Number of pushing MTs', str(len(spindle_from_dict.push_state[spindle_from_dict.push_state != 0]))) # number of pushing MTs
    #     outer_table.add_row('Number of pulling MTs', str(len(spindle_from_dict.pull_state[spindle_from_dict.pull_state != 0]))) # number of pulling MTs
    #     outer_table.add_row('Tubulin use / tubulin budget (um)', f'{spindle_from_dict.calculate_tubulin_use()} / {spindle_from_dict.tubulin_budget}') # tubulin use / tubulin budget
    #     outer_table.add_row('Attempt Counter', str(spindle_from_dict.num_attempts)) # attempt counter
    #     outer_table.add_row('Number of Positions Accepted', str(spindle_from_dict.num_accepted_states)) # number of accepted positions
    #     live.update(outer_table)

    # # -- reconstructing trajectory from spindle trace folder --

    # spindle_trace_dir = os.path.join(experiment_dir, 'spindle_trace')

    # # states are saved as batched arrays of shape (batch, num_sites) named
    # # 'push_states_{start}_{end}.npy' / 'pull_states_{start}_{end}.npy'. Sort the batches
    # # by their start index so we replay accepted states in the order they were produced.
    # def batch_start(path):
    #     return int(os.path.basename(path).split('_')[-2])

    # push_batch_paths = sorted(glob.glob(os.path.join(spindle_trace_dir, 'push_states_*.npy')), key=batch_start)

    # state_index = 0
    # done = False
    # with Live(console=console, refresh_per_second=4) as live:
    #     for push_batch_path in push_batch_paths:
    #         if done:
    #             break
    #         # pair each push batch with its pull batch by the shared {start}_{end} suffix
    #         pull_batch_path = os.path.join(spindle_trace_dir, os.path.basename(push_batch_path).replace('push_states_', 'pull_states_', 1))

    #         push_batch = np.load(push_batch_path)
    #         pull_batch = np.load(pull_batch_path)

    #         for new_push_state, new_pull_state in zip(push_batch, pull_batch):
    #             # load the accepted state onto the spindle
    #             spindle_from_dict.push_state = new_push_state
    #             spindle_from_dict.pull_state = new_pull_state

    #             # evolve time
    #             new_mtoc_positions, boundary_violated, trajectory  = spindle_from_dict.time_evolution()

    #             # set new MTOC positions
    #             spindle_from_dict.mtoc_positions = new_mtoc_positions
    #             cost = spindle_from_dict.calculate_cost()
    #             spindle_from_dict.time = spindle_from_dict.time + spindle_from_dict.evolution_time
    #             # fold in trajectory. This is an empty dictionary if spindle_from_dict.save_trajectory == False
    #             spindle_from_dict.trajectory = spindle_from_dict.trajectory | trajectory

    #             # -- readout --

    #             outer_table = Table(title="Restarting Spindle Simulation")
    #             outer_table.add_column("Parameter", justify="left")
    #             outer_table.add_column("Value", justify="right")

    #             # set table values
    #             outer_table.add_row(f'State', f'{state_index} / {num_accepted_states}') # state number
    #             outer_table.add_row('Current Time', str(spindle_from_dict.time)) # last stable time
    #             outer_table.add_row('current_cost', str(cost)) # last accepted cost
    #             outer_table.add_row('Current Position', str(spindle_from_dict.mtoc_positions)) # last stable time
    #             outer_table.add_row('number of MTs', str(spindle_from_dict.calculate_num_mts())) # number of MTs
    #             outer_table.add_row('tubulin use / tubulin budget', f'{spindle_from_dict.calculate_tubulin_use()} / {spindle_from_dict.tubulin_budget}') # tubulin use / tubulin budget
    #             proc_rss_gb = psutil.Process().memory_info().rss / 1e9
    #             slurm_mb = os.environ.get("SLURM_MEM_PER_NODE")
    #             mem_str = f"{proc_rss_gb:.2f} GB"
    #             if slurm_mb:
    #                 mem_str += f" / {int(slurm_mb)/1e3:.1f} GB alloc"
    #             outer_table.add_row('Memory (RSS)', mem_str)
    #             live.update(outer_table)

    #             state_index += 1
    #             if max_states is not None and state_index >= max_states:
    #                 done = True
    #                 break

    return spindle_from_dict
    

class Spindle:

    def __init__(
            self, 
            initial_mtoc_positions, # numpy array of shape (k,3) for k MTOCs 
            push_lattice,
            pull_lattice,
            initial_time=0.0,

            # -- optimization paramters --
            tubulin_budget=100.0, # µm
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
            growth_rate=1.0, # µm s^{−1}
            pull_force=5.0, # pN 
            stall_force=5.0, # pN
            cytoplasmic_drag_factor=100.0, # pN s µm^{−1} drag factor of aster
            boundary_radius=10.0, # µm
            motor_radius=1.0, # µm
            average_mt_length=10, # µm
            spindle_length=13, # µm

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

            seed                  (int or None): Random seed for reproducibility.
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
        self.num_attempts = num_attempts
        self.num_accepted_states = num_accepted_states

        # -- setting hyperparameters --
        self.euler_timestep_size = euler_timestep_size
        self.evolution_time = evolution_time 
        # self.cost_tolerance = cost_tolerance 
        self.optimization_temperature = optimization_temperature
        
        # -- setting parameters --
        self.time = initial_time
        
        # initialize lattices
        self.push_lattice = push_lattice
        self.pull_lattice = pull_lattice

        # initialize spindle states:
        # this uses the convention where push_lattice[i] = 0 means the site is empty 
        # and push_lattice[j]=k means that there is an MT from aster k impinging on site j
        # similar for push_lattice
        self.num_push_sites = push_lattice.shape[0]
        self.num_pull_sites = pull_lattice.shape[0]

        self.push_state = np.zeros(self.num_push_sites)
        self.pull_state = np.zeros(self.num_pull_sites)

        self.push_boundary_unit_normals = normalize_vecs(push_lattice)[0] # only matters for calculating pushing force magnitude

        # initialize position of MTOCs

        mtoc_positions = {}

        num_asters = len(initial_mtoc_positions)
        for aster in range(num_asters):
            mtoc_positions[aster+1] = initial_mtoc_positions[aster]
        
        self.mtoc_positions = mtoc_positions

        # set up map from pushing lattice to pulling lattice 
        tree = cKDTree(pull_lattice)

        # For every high-res point, find its nearest low-res neighbor + distance
        distances, nearest_pull_idx = tree.query(push_lattice, k=1)

        push_to_pull = { # high res lattice to low res lattice
            push_idx: pull_idx
            for push_idx, (pull_idx, dist) in enumerate(zip(nearest_pull_idx, distances)) # find the nearest pushing point for each pull point 
            if dist <= motor_radius # ensure we only keep push points within motor radius to the nearest pull point
        }

        self.push_to_pull = push_to_pull # many fewer than in the original pushing lattice 

        # -- setting up data saving -- 

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.path_to_data = data_dir # path to the parent directory within which we make the director for each experiment

        self.save = save

        if dir_path is None and save: # meaning a directory for the experiment has not already been made
            if dir_prefix == '':
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                dir_prefix = f'{num_asters}_asters_{tubulin_budget}_um_tubulin_{boundary_radius}_um_radius_{timestamp}'
            dirname = f'{dir_prefix}'
            self.dir_path = os.path.join(data_dir, dirname)
            os.makedirs(self.dir_path, exist_ok=True) # raises error if data directory does not exist

            # make folder for plots
            self.plot_folder_path = os.path.join(self.dir_path, 'plots')
            os.makedirs(self.plot_folder_path, exist_ok=True)

            # make spindle_trace folder
            self.spindle_trace_path = os.path.join(self.dir_path, 'spindle_trace')
            os.makedirs(self.spindle_trace_path, exist_ok=True) # raises error if data directory does not exist

            # save spindle params as dict
            spindle_dict = dict(self.__dict__)

            spindle_path = os.path.join(self.dir_path, 'spindle.pkl')
            with open(spindle_path, "wb") as f:
                pickle.dump(spindle_dict, f)
            print(f'spindle saved to {spindle_path}')

            # save num_accepted_states
            np.save(os.path.join(self.dir_path, 'num_accepted_states.npy'), self.num_accepted_states)

            # save num_total_attempts
            np.save(os.path.join(self.dir_path, 'num_attempts.npy'), self.num_attempts)

        else:
            self.dir_path = dir_path
            self.plot_folder_path = os.path.join(self.dir_path, 'plots')
            self.spindle_trace_path = os.path.join(self.dir_path, 'spindle_trace')

        self.save_trajectory = save_trajectory
        # if save_trajectory:
        self.trajectory = trajectory
        trajectory_path = os.path.join(self.dir_path, 'trajectory.pkl')
        with open(trajectory_path, "wb") as f:
            pickle.dump(self.trajectory, f)


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
        connected_points = self.pull_lattice[self.pull_state == mtoc_id]

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
        connected_points = self.push_lattice[self.push_state == mtoc_id]
        relevant_mt_vecs = connected_points - self.mtoc_positions[mtoc_id]
        dirs, norms = normalize_vecs(relevant_mt_vecs)

        # -- calculating buckling forces --
        # calculating the effective force coefficients (mt_dir . boundary_norm)
        buckling_forces = (np.pi**2) * self.rigidity / (norms**2)

        # -- calculating unbuckled pushing forces --
        # calculating the effective force coefficients (mt_dir . boundary_norm)
        pushing_boundary_normals = self.push_boundary_unit_normals[self.push_state == mtoc_id]
        effective_force_coefficients = np.sum(dirs * pushing_boundary_normals, axis=1)
        # calculating the denominator of the pushing force magnitude
        pushing_force_denominators = (self.stall_force / (self.growth_rate * self.sliding_friction_coefficient)) * (1 - effective_force_coefficients) + 1
        # putting the pieces together
        pushing_force_magnitudes = self.stall_force / pushing_force_denominators

        # -- pushing forces are bounded above by the buckling force --
        pushing_force_magnitudes[pushing_force_magnitudes > buckling_forces] = buckling_forces[pushing_force_magnitudes > buckling_forces]

        # total pushing force is the component-wise sum of the pushing vectors
        pushing_vectors = pushing_force_magnitudes[:, np.newaxis] * dirs

        # -- summing over pushing force vectors to find total pushing force --
        # pushing_mt_dirs point outwards from the mtoc, we want pushing forces to point inwards towards the mtoc
        total_pushing_force = -np.sum(pushing_vectors, axis=0)

        return total_pushing_force


    def calc_mtoc_velocity(self, mtoc_id):
        """Calculates the velocity of an mtoc based on the mtoc position and the set of pushing and pulling mts connected to it.
        This is the implementation of the mtoc's equation of motion.

        Returns:
            numpy.array: velocity vector in the form of numpy.array([x,y,z])
        """
        if mtoc_id not in self.mtoc_positions.keys():
            raise ValueError("the provided mtoc_id does not refer to an mtoc in this spindle")
        
        return (self.calculate_pulling_forces(mtoc_id) + self.calculate_pushing_forces(mtoc_id)) / self.cytoplasmic_drag_factor


    def time_evolution(self, evolution_time=None):
        """evolve time in the system by some amount.
        Progresses time for all MTOCs.

        Args:
            time_to_evolve_by (float): amount of time to evolve by in seconds

        Returns:
            dict, dict: MTOC positions after time evolution, trajectory of evolution
        """
        if evolution_time is None:
            evolution_time = self.evolution_time
        
        time_before_evolution = np.copy(self.time)
        boundary_violated = False

        current_time = float(np.copy(self.time))

        mtoc_positions = self.mtoc_positions.copy()

        save_trajectory = self.save_trajectory
        trajectory = {}

        
        while (current_time - time_before_evolution) < evolution_time and not boundary_violated:

            # evolve each MTOC by one timestep.
            for mtoc_id in self.mtoc_positions.keys():
                # calculate velocity
                dr_dt = self.calc_mtoc_velocity(mtoc_id) 
                
                # calculate the new position of the MTOC
                mtoc_positions[mtoc_id] = mtoc_positions[mtoc_id] + (dr_dt * self.euler_timestep_size)

                # check that the new mtoc position is not outside of the radius
                normalized_new_mtoc_pos, new_mtoc_pos_norm = normalize_vecs(mtoc_positions[mtoc_id])
                if new_mtoc_pos_norm > self.boundary_radius:
                    boundary_violated = True

            current_time += self.euler_timestep_size

            if save_trajectory:
                timepoint_data = {
                    # 'push_state': self.push_state.astype(int),
                    # 'pull_state': self.pull_state.astype(int),
                    'mtoc_pos': mtoc_positions,
                    'boundary_violated': boundary_violated,
                    'cost': self.calculate_cost(),
                    'tubulin_use': self.calculate_tubulin_use(),
                    'num_mts': self.calculate_num_mts(),
                }
                trajectory[current_time] = timepoint_data
        
        return mtoc_positions, boundary_violated, trajectory,
    
    def calculate_tubulin_use(self):
        total_mt_length = 0
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
            
            
            # add the sums of these norms to the total_mt_length
            total_mt_length += np.sum(push_norms) + np.sum(pull_norms)

        return total_mt_length
    

    def calculate_num_mts(self):

        push_mts = np.where(self.push_state != 0)[0]
        pull_mts = np.where(self.pull_state != 0)[0]

        num_mts = len(push_mts) + len(pull_mts)
        return num_mts


    def calculate_cost(self):
        """Calculates cost
        This cost function has three term types: 
        1. a term penalizing the over or under use of tubulin;
        2. a collection of terms saying that each MTOC wants to be in the centre of the sphere;
        3. a collection of terms saying that each pair of MTOCs wants to be as far as possible from all other MTOCs.

        Returns:
            float: cost
        """
        spatial_coefficient = 1000
        material_coefficient = 1/2
        
        cost = 0
        mtoc_ids = np.array(list(self.mtoc_positions.keys()))

        # --- explicitly set MTOC positions ---
        # single aster set to a particular position
        # desired_mtoc_positions = {1: np.array([5.0, 0, 0])}

        # cost += spatial_coefficient * normalize_vecs(self.mtoc_positions[1] - desired_mtoc_positions[1])[1] / self.boundary_radius
        # --- ---

        # --- spatial cost ---

        # # -- desired spindle elongation --
        current_spindle_length = 0
        i, j = np.triu_indices(len(mtoc_ids), k=1)  # k=1 excludes diagonal (no (i,i) pairs)
        pairs = np.column_stack([mtoc_ids[i], mtoc_ids[j]])

        for pair in pairs:
            current_spindle_length += normalize_vecs((self.mtoc_positions[pair[0]] - self.mtoc_positions[pair[1]]))[1]
        current_spindle_length = current_spindle_length / len(pairs)

        # for two asters
        # two_aster_spindle_length = normalize_vecs(self.mtoc_positions[1] - self.mtoc_positions[2])[1]
        # current_spindle_length = two_aster_spindle_length

        cost += spatial_coefficient * np.square(1 - (current_spindle_length / self.spindle_length))

        # # -- spindle centring -- 
        # place the centre of the spindle at the origin
        # sum_positions = np.zeros(3)
        # for i in range(len(mtoc_ids)):
        #     sum_positions += self.mtoc_positions[mtoc_ids[i]]
        
        # cost += np.square(normalize_vecs(sum_positions)[1] / (self.boundary_radius))

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
        cost += material_coefficient * np.abs(1 - (total_mt_length / self.tubulin_budget))

        # --- --- 
        # if cost < 0.01:
        #     cost = 0

        return cost
        

    def sample_spindle_update(self):
        
        # choose whether to add or remove an MT
        add = self.rng.choice([True, False], p=[0.5, 0.5]) # True -> add, False -> remove

        if add:

            # choose an empty site to place an MT
            empty_sites = np.where(self.push_state == 0)[0]

            if empty_sites.size == 0: # avoiding self.rng errors
                push = True
                state = self.push_state
                return push, state
            
            # choose which mtoc we are nucleating from
            mtoc_id = self.rng.choice(list(self.mtoc_positions.keys()))

            # -- spatially uniform sampling --
            site_to_fill = self.rng.choice(empty_sites)
            # -- -- 

            # # -- exponential length distributed sampling --

            # empty_site_distances = normalize_vecs(self.push_lattice[empty_sites] - self.mtoc_positions[mtoc_id])[1] # norms of difference vectors
            # length_probability = np.exp((-empty_site_distances / self.average_mt_length)) # calculate probabilities of an MT growing to be at least that long
            # site_selection_probabilities = length_probability / np.sum(length_probability) # normalize

            # site_to_fill = self.rng.choice(empty_sites, p=site_selection_probabilities)  # choose
            # # -- --

            # -- push or pull
            # if site_to_fill is within the capture radius
            push = True
            if site_to_fill in self.push_to_pull.keys():
                if self.pull_state[self.push_to_pull[site_to_fill]] == 0:
                    push = False
                    site_to_fill = self.push_to_pull[site_to_fill]

            if push:
                state = self.push_state.copy()
            else:
                state = self.pull_state.copy()

            state[site_to_fill] = mtoc_id
            
            # # -- choose which MTOC this MT is nucleating from uniformly --
            # mtoc_ids = np.array(list(self.mtoc_positions.keys()))
            # mtoc_to_nucleate_from = self.rng.choice(mtoc_ids)
            # state[site_to_fill] = mtoc_to_nucleate_from
        else: # remove
            # choose uniformly which MT to remove
            # choose to remove from push or pull weighted by the proportion of MTs which are pushing or pulling
            num_pushing = len(self.push_state[self.push_state != 0])
            num_pulling = len(self.pull_state[self.pull_state != 0])

            numerator = num_pushing + num_pulling
            if numerator == 0:
                numerator = 1

            p_push = num_pushing / numerator
            push = self.rng.choice([True, False], p=[p_push, 1-p_push]) # True -> add, False -> remove

            if push:
                state = self.push_state.copy()
            else:
                state = self.pull_state.copy()

            # choose a filled site to empty 
            filled_sites = np.where(state != 0)[0]

            if filled_sites.size == 0: # avoiding self.rng errors
                return push, state

            # choose from uniform distribution
            site_to_empty = self.rng.choice(filled_sites)

            # empty sites are set to 0
            state[site_to_empty] = 0
        
        return push, state


    def optimize(self, total_attempts, save_batch_size=1000):

        initial_cost = len(self.push_state) # setting initial cost to be very high
        old_mtoc_positions = self.mtoc_positions.copy() # initial original state is the current state
        old_cost = initial_cost # any stable position is an improvement
        old_time = np.copy(self.time)
        old_push_state = np.copy(self.push_state)
        old_pull_state = np.copy(self.pull_state)

        # states will be saved every 1000 accepted states.
        # accepted_states is a list where each element is the tuple (push_state, pull_state)
        accepted_pull_states = []
        accepted_push_states = []
        num_accepted_states_at_empty = np.copy(self.num_accepted_states)

        def flush_accepted_states():
            """Persist the accumulated accepted states as a batch and reset the buffers."""
            nonlocal accepted_push_states, accepted_pull_states, num_accepted_states_at_empty
            if not accepted_push_states:
                return

            start = num_accepted_states_at_empty
            end = self.num_accepted_states
            np.save(os.path.join(self.spindle_trace_path, trace_batch_name('push', start, end)), np.array(accepted_push_states))
            np.save(os.path.join(self.spindle_trace_path, trace_batch_name('pull', start, end)), np.array(accepted_pull_states))

            accepted_push_states = []
            accepted_pull_states = []
            num_accepted_states_at_empty = np.copy(self.num_accepted_states)

            # keep the experiment-level counters in sync with what's on disk
            np.save(os.path.join(self.dir_path, 'num_accepted_states.npy'), self.num_accepted_states)
            np.save(os.path.join(self.dir_path, 'num_attempts.npy'), self.num_attempts)
            
            # save trajectory
            trajectory_path = os.path.join(self.dir_path, 'trajectory.pkl')
            safe_pickle_dump(self.trajectory, trajectory_path)
            # with open(trajectory_path, "wb") as f:
            #     pickle.dump(self.trajectory, f)
                

        # -- wall time readout --
        start = time.perf_counter()

        def format_elapsed(start):
            elapsed = time.perf_counter() - start
            return f"{elapsed:.2f}s"
        
        with Live(console=console, refresh_per_second=4) as live:

            # --- update spindle -> relax loop ---

            while self.num_attempts <= total_attempts:

                # -- middle loop (update spindle, try to relax, if cost improves, accept, if not, reject) -- 
                acceptable = False
                attempt_counter = 0
                
                while not acceptable and self.num_attempts <= total_attempts:

                    # -- readout -- 
                    # initialize table
                    outer_table = Table(title="Spindle Simulation")
                    outer_table.add_column("Parameter", justify="left")
                    outer_table.add_column("Value", justify="right")
                    
                    # set table values
                    outer_table.add_row('Wall elapsed time (s)', format_elapsed(start))
                    outer_table.add_row('Last Accepted Time (s)', str(old_time)) # last stable time
                    outer_table.add_row('Last Accepted Position (um)', str(old_mtoc_positions)) # last stable position
                    outer_table.add_row('Spindle Length (um)', f'{normalize_vecs(self.mtoc_positions[1] - self.mtoc_positions[2])[1]} / {self.spindle_length}') # tubulin use / tubulin budget
                    outer_table.add_row('Net Force on aster 1 (pN)', f'{self.calc_mtoc_velocity(1)*self.cytoplasmic_drag_factor}') # force vector on MTOC
                    outer_table.add_row('Last Accepted Cost', str(old_cost)) # last accepted cost
                    outer_table.add_row('Number of MTs', str(self.calculate_num_mts())) # number of MTs
                    outer_table.add_row('Number of pushing MTs', str(len(self.push_state[self.push_state != 0]))) # number of pushing MTs
                    outer_table.add_row('Number of pulling MTs', str(len(self.pull_state[self.pull_state != 0]))) # number of pulling MTs
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

                    # update meta-spindle-state
                    push, new_state = self.sample_spindle_update()

                    if push:
                        self.push_state = new_state
                    else:
                        self.pull_state = new_state

                    attempt_counter += 1
                    self.num_attempts +=1

                    # evolve time for metastate
                    new_mtoc_positions, meta_boundary_violated, meta_trajectory  = self.time_evolution()
                    
                    # set new mtoc_positions
                    self.mtoc_positions = new_mtoc_positions

                    meta_cost = self.calculate_cost()
                    meta_time = self.time + self.evolution_time

                    # -- simulated annealing --
                    
                    # states which violated the boundary are always rejected
                    # otherwise, we follow Metropolis-Hastings style simulated annealing
                    if meta_cost < old_cost and not meta_boundary_violated:
                        acceptable = True # accept strict decreases in cost
                    elif not meta_boundary_violated:
                        # it must be true that meta_cost >= old_cost
                        cost_delta = meta_cost - old_cost # > 0
                        prob_of_acceptance = np.exp(-cost_delta / self.optimization_temperature)

                        acceptable = self.rng.choice([True, False], p=[prob_of_acceptance, 1-prob_of_acceptance])

                    # reverse metastate if the spindle modification is not acceptable
                    if not acceptable:
                        # reset spindle states
                        if push:
                            self.push_state = old_push_state
                        else:
                            self.pull_state = old_pull_state
                        # reset mtoc positions
                        self.mtoc_positions = old_mtoc_positions

                # -- back in outer loop, saving new accepted position -- 
                # current mtoc positions and spindle states become old mtoc positions and spindle states
                old_mtoc_positions = self.mtoc_positions.copy() # this is the current metastate position
                old_cost = meta_cost
                old_time = meta_time
                self.time = meta_time
                if self.save:
                    accepted_push_states.append(old_push_state)
                    accepted_pull_states.append(old_pull_state)
                old_push_state = np.copy(self.push_state)
                old_pull_state = np.copy(self.pull_state)
                
                self.num_accepted_states += 1
                attempt_counter = 0

                # fold in trajectory. This is an empty dictionary if self.save_trajectory == False
                # self.trajectory = self.trajectory | meta_trajectory
                if self.save_trajectory:
                    total_force = {}
                    for mtoc_id in self.mtoc_positions.keys():
                        total_force[mtoc_id] = self.cytoplasmic_drag_factor * self.calc_mtoc_velocity(mtoc_id)
                    timepoint_data = {
                        # 'push_state': self.push_state.astype(int),
                        # 'pull_state': self.pull_state.astype(int),
                        'mtoc_pos': self.mtoc_positions,
                        'cost': self.calculate_cost(),
                        'tubulin_use': self.calculate_tubulin_use(),
                        'num_mts': self.calculate_num_mts(),
                        'total_force': total_force,
                    }
                    self.trajectory[self.time] = timepoint_data


                if self.save and (self.num_accepted_states - num_accepted_states_at_empty) >= save_batch_size:
                    # save the last batch of accepted states to the spindle_trace folder
                    # trajectory is only used for making animations.
                    flush_accepted_states()

        # persist any trailing accepted states that didn't fill a final 1000-state batch
        # if self.save:
        #     flush_accepted_states()

    def plot_cost(self):
        times = list(self.trajectory.keys())

        costs = []
        for time in times:
            costs.append(self.trajectory[time]['cost'])

        plt.plot(times, costs)
        plt.xlabel('time (s)')
        plt.ylabel('cost (dimensionless)')
        plt.title(f'2_aster_{self.tubulin_budget}_tubulin_{self.optimization_temperature}_temp')
        # plt.show()
        ceph = '/mnt/home/ealca/ceph/multi-aster/push_and_pull_test/'

        # self.plot_folder_path = os.path.join(ceph, 'plots')
        # os.makedirs(self.plot_folder_path, exist_ok=True)

        cost_plot_path = os.path.join(self.plot_folder_path, f'cost_time_plot_2_aster_{self.tubulin_budget}_tubulin_{self.optimization_temperature}_temp.png')
        plt.savefig(cost_plot_path)
        print(f'cost vs time plot saved to {cost_plot_path}')
        plt.close()

    
    def plot_aster_separation(self):

        times = list(self.trajectory.keys())

        norm1 = []
        norm2 = []
        aster_separations = []
        for time in times:
            mtoc_positions = self.trajectory[time]['mtoc_pos']
            norm1.append(normalize_vecs(mtoc_positions[1])[1])
            norm2.append(normalize_vecs(mtoc_positions[2])[1])
            aster_separations.append(normalize_vecs(mtoc_positions[1] - mtoc_positions[2])[1])


        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # plotting aster norms
        ax1.plot(times, norm1, label='aster 1 distance from centre')
        ax1.plot(times, norm2, label='aster 2 distance from centre')
        ax1.set_ylabel('Distance (um)')
        ax1.set_ylim([0, self.boundary_radius])
        ax1.legend()

        # plotting separation
        ax2.plot(times, aster_separations, label='distance between asters')
        ax2.set_ylim([0, 2 * self.boundary_radius])
        ax2.set_ylabel('aster separation (um)')
        ax2.set_xlabel('time (s)')

        # plotting spindle deviation from origin (R1-R2)

        fig.tight_layout()

        # aster_norm_plot_path = os.path.join(spindle.plot_folder_path, f'aster_norms.png')
        # aster_sep_plot_path = os.path.join(spindle.plot_folder_path, f'aster_separation.png')
        combined_plot_path = os.path.join(self.plot_folder_path, f'aster_dynamics.png')
        plt.savefig(combined_plot_path)
        plt.close()

        ani_path = os.path.join(self.plot_folder_path, f'ani.mp4')
        self.animate_mtoc_trajectory(save_path=ani_path)

    def calculate_motor_occupancy(self, mtoc_id, start_time, end_time):

        start_time = int(np.round(start_time, 4) / self.evolution_time)
        end_time = int(np.round(end_time, 4) / self.evolution_time)
        
        spindle_trace_path = os.path.join(self.dir_path, 'spindle_trace')
        
        # --- isolate the spindle states between start_time and end_time --- 

        # find the files and load their data
        top = (np.ceil(end_time / 1000) * 1000).astype(int) # nearest thousand greater than end_time
        bottom = (np.floor(start_time / 1000) * 1000).astype(int) # nearest thousand less than end_time

        temp_top = bottom + 1000
        temp_bottom = np.copy(bottom)
        
        temp_spindle_states = []
        
        while temp_top <= top:
            next_states_path = os.path.join(spindle_trace_path, f'pull_states_{temp_bottom}_{temp_top}.npy')
            temp_spindle_states.extend(np.load(next_states_path))
            # print(next_states_path)
            temp_bottom += 1000
            temp_top +=1000
        temp_spindle_states = np.array(temp_spindle_states)
        # print(temp_spindle_states.shape)

        # -- cull this big set of spindle states to include only states between start_time and end_time --

        # find the number of states to cull from the start
        start_time_discrepancy = start_time - bottom
        end_time_discrepancy = top - end_time

        spindle_states_between_start_end = temp_spindle_states[start_time_discrepancy:]
        if end_time_discrepancy > 0:
            spindle_states_between_start_end = spindle_states_between_start_end[:-end_time_discrepancy]

        # -- calculate occupancy --
        
        occupancy = np.zeros_like(spindle_states_between_start_end[0])
        # indices = np.arange(len(occupancy)
        
        for state in spindle_states_between_start_end:
            occupancy += (state == mtoc_id).astype(int)
        
        occupancy = occupancy / (end_time - start_time) # proportion of time motor is filled
        return occupancy

    def animate_mtoc_trajectory(self, save_path=None, interval=50, stride=100, show_occupancy=False, occupancy_mtoc_id=1):
        print(list(self.trajectory.keys())[-1])
        times = sorted(list(self.trajectory.keys())[:-1000])[::stride]
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

        # precompute per-frame occupancy
        occupancy_frames = None
        occ_scatter = None
        if show_occupancy:
            window = stride * self.evolution_time
            last_time = sorted(self.trajectory.keys())[-1]
            print(f'Precomputing occupancy for {len(times)} frames...')
            occupancy_frames = []
            for i, t in enumerate(times):
                t_end = t + window
                if t_end > last_time:
                    if occupancy_frames:
                        occupancy_frames.append(occupancy_frames[-1])
                        continue
                    t_end = last_time
                occupancy_frames.append(self.calculate_motor_occupancy(occupancy_mtoc_id, t, t_end))
            occupancy_frames = np.array(occupancy_frames)
            print('Done.')

            occ_norm = mcolors.Normalize(vmin=occupancy_frames.min(), vmax=max(occupancy_frames.max(), 1e-9))
            lx, ly, lz = self.pull_lattice[:, 0], self.pull_lattice[:, 1], self.pull_lattice[:, 2]
            occ_scatter = ax.scatter(lx, ly, lz, c=occupancy_frames[0], cmap='inferno', norm=occ_norm, alpha=0.5, s=10)
            fig.colorbar(occ_scatter, ax=ax, shrink=0.5, label=f'Occupancy (MTOC {occupancy_mtoc_id})')

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
        ax.legend()

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
            -0.1, 0.7,
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
            if occ_scatter is not None:
                occ_scatter.set_array(occupancy_frames[frame])
            cost, tubulin_use, num_mts, dist, forces = _stats(t)
            info_text.set_text(_format_text(t, cost, tubulin_use, num_mts, dist, forces))
            return scatters + ([occ_scatter] if occ_scatter is not None else []) + [info_text]

        anim = FuncAnimation(fig, update, frames=len(times), interval=interval, blit=False)

        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=1000 // interval)
            print(f'animation saved to {save_path}')
            plt.close()
        else:
            plt.show()

        return anim