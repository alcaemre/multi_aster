#
# Emre Alca
# University of Pennsylvania
# Created on Wed Jun 03 2026
# Last Modified: 2026/07/29 16:04:09
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

    push_state = np.zeros(spindle_dict['push_lattice'].shape[0])
    pull_state = np.zeros(spindle_dict['pull_lattice'].shape[0])

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

    # reconstruct trajectory

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
    
    spindle_from_dict.push_state = push_state
    spindle_from_dict.pull_state = pull_state

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
            safe_pickle_dump(spindle_dict, spindle_path)
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
        safe_pickle_dump(self.trajectory, trajectory_path)


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
        spatial_coefficient = 100
        material_coefficient = 1
        centring_coefficient = 1
        no_net_force_coefficient = 1
        
        cost = 0
        mtoc_ids = np.array(list(self.mtoc_positions.keys()))

        # --- explicitly set MTOC positions ---
        # single aster set to a particular position
        # desired_mtoc_positions = {1: np.array([5.0, 0, 0])}

        # cost += spatial_coefficient * normalize_vecs(self.mtoc_positions[1] - desired_mtoc_positions[1])[1] / self.boundary_radius
        # --- ---

        # --- spatial cost ---

        # # -- desired spindle elongation --
        # current_spindle_length = 0
        # i, j = np.triu_indices(len(mtoc_ids), k=1)  # k=1 excludes diagonal (no (i,i) pairs)
        # pairs = np.column_stack([mtoc_ids[i], mtoc_ids[j]])

        # for pair in pairs:
        #     current_spindle_length += normalize_vecs((self.mtoc_positions[pair[0]] - self.mtoc_positions[pair[1]]))[1]
        # current_spindle_length = current_spindle_length / len(pairs)

        # for two asters
        # two_aster_spindle_length = normalize_vecs(self.mtoc_positions[1] - self.mtoc_positions[2])[1]
        # current_spindle_length = two_aster_spindle_length

        # cost += spatial_coefficient * np.square(1 - (current_spindle_length / self.spindle_length))

        # # -- spindle centring -- 
        # place the centre of the spindle at the origin
        # sum_positions = np.zeros(3)
        # for i in range(len(mtoc_ids)):
        #     sum_positions += self.mtoc_positions[mtoc_ids[i]]
        
        # cost += centring_coefficient * np.square(normalize_vecs(sum_positions)[1] / (self.boundary_radius))

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
        cost += material_coefficient * np.abs(1 - (total_mt_length / self.tubulin_budget))

        # --- --- 
        # if cost < 0.01:
        #     cost = 0

        return cost
        

    def sample_spindle_update(self, add=None, mtoc_id=None):

        # we want to choose push or pull, 
        
        # choose whether to add or remove an MT
        if add is None:
            add = self.rng.choice([True, False], p=[0.5, 0.5]) # True -> add, False -> remove

        if add:

            # choose an empty site to place an MT
            empty_sites = np.where(self.push_state == 0)[0]

            if empty_sites.size == 0: # avoiding self.rng errors
                push = True
                state = self.push_state
                return push, 0, 0
            
            # choose which mtoc we are nucleating from
            if mtoc_id is None:
                mtoc_id = self.rng.choice(list(self.mtoc_positions.keys()))

            # -- spatially uniform sampling --
            # site_to_fill = self.rng.choice(empty_sites)
            # -- -- 

            # # -- exponential length distributed sampling --

            empty_site_distances = normalize_vecs(self.push_lattice[empty_sites] - self.mtoc_positions[mtoc_id])[1] # norms of difference vectors
            length_probability = np.exp((-empty_site_distances / self.average_mt_length)) # calculate probabilities of an MT growing to be at least that long
            site_selection_probabilities = length_probability / np.sum(length_probability) # normalize

            site_to_fill = self.rng.choice(empty_sites, p=site_selection_probabilities)  # choose
            # # -- --

            # -- push or pull
            # if site_to_fill is within the capture radius
            push = True
            if site_to_fill in self.push_to_pull.keys():
                if self.pull_state[self.push_to_pull[site_to_fill]] == 0:
                    push = False
                    site_to_fill = self.push_to_pull[site_to_fill]

            # if push:
            #     state = self.push_state.copy()
            # else:
            #     state = self.pull_state.copy()

            # state[site_to_fill] = mtoc_id

            lattice_site = site_to_fill
            site_value = mtoc_id
            
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
                return push, 0, 0

            # choose from uniform distribution
            site_to_empty = self.rng.choice(filled_sites)

            # empty sites are set to 0
            state[site_to_empty] = 0

            lattice_site = site_to_empty
            site_value = 0

        return push, lattice_site, site_value
        
        # return push, state


    def optimize(self, total_attempts, save_batch_size=1000, max_lab_time=None):
        """
        Run the optimize loop until self.num_attempts exceeds total_attempts
        or, if max_lab_time (seconds of simulated/lab time) is given, until
        self.time exceeds it -- whichever happens first.
        """

        initial_cost = len(self.push_state) # setting initial cost to be very high
        old_mtoc_positions = self.mtoc_positions.copy() # initial original state is the current state
        old_cost = initial_cost # any stable position is an improvement
        old_time = np.copy(self.time)
        # trace will be saved every save_batch_size accepted states.
        # the spindle_trace tracks the changes in the spindle
        spindle_trace = []

        # # accepted_states is a list where each element is the tuple (push_state, pull_state)
        # accepted_pull_states = []
        # accepted_push_states = []
        num_accepted_states_at_empty = np.copy(self.num_accepted_states)

        def flush_accepted_states():
            """Persist the accumulated accepted states as a batch and reset the buffers."""
            # nonlocal accepted_push_states, accepted_pull_states, num_accepted_states_at_empty
            nonlocal spindle_trace, num_accepted_states_at_empty
            if not spindle_trace:
                return

            start = num_accepted_states_at_empty
            end = self.num_accepted_states
            np.save(os.path.join(self.spindle_trace_path, f'spindle_trace_{start}_{end}.npy' ), np.array(spindle_trace))
            # np.save(os.path.join(self.spindle_trace_path, trace_batch_name('push', start, end)), np.array(accepted_push_states))
            # np.save(os.path.join(self.spindle_trace_path, trace_batch_name('pull', start, end)), np.array(accepted_pull_states))

            # accepted_push_states = []
            # accepted_pull_states = []
            spindle_trace = []
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

        def lab_time_exceeded():
            return max_lab_time is not None and self.time >= max_lab_time

        def keep_going():
            return self.num_attempts <= total_attempts and not lab_time_exceeded()

        with Live(console=console, refresh_per_second=4) as live:

            # --- update spindle -> relax loop ---

            while keep_going():

                # -- middle loop (update spindle, try to relax, if cost improves, accept, if not, reject) --
                acceptable = False
                attempt_counter = 0

                while not acceptable and keep_going():

                    # -- readout -- 
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
                    # push, new_state = self.sample_spindle_update()
                    push, lattice_site, site_value = self.sample_spindle_update()

                    if push:
                        old_site_value = self.push_state[lattice_site]
                        self.push_state[lattice_site] = site_value
                    else:
                        old_site_value = self.pull_state[lattice_site]
                        self.pull_state[lattice_site] = site_value

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
                        # restore only the single element that was changed
                        if push:
                            self.push_state[lattice_site] = old_site_value
                        else:
                            self.pull_state[lattice_site] = old_site_value
                        # reset mtoc positions
                        self.mtoc_positions = old_mtoc_positions

                # -- back in outer loop, saving new accepted position -- 
                # current mtoc positions and spindle states become old mtoc positions and spindle states
                old_mtoc_positions = self.mtoc_positions.copy() # this is the current metastate position
                old_cost = meta_cost
                old_time = meta_time
                self.time = meta_time
                if self.save:
                    spindle_trace.append((push, lattice_site, site_value))

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
        ax2.set_ylim([0, 2 * self.boundary_radius])
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
            ax.plot(times, distances[mtoc_id], color=colors[i % len(colors)], label=f'aster {mtoc_id} distance from centre')
        ax.set_ylabel('Distance (um)')
        ax.set_ylim([0, self.boundary_radius])
        ax.set_xlabel('time (s)')
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(self.plot_folder_path, 'aster_distance_from_centre.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'aster distance from centre saved to {plot_path}')

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
        spindle_trace_dir = os.path.join(self.dir_path, 'spindle_trace')
        trace_files = sorted(
            glob.glob(os.path.join(spindle_trace_dir, 'spindle_trace_*.npy')),
            key=lambda f: int(os.path.basename(f).split('_')[2])
        )

        mtoc_ids = sorted(self.mtoc_positions.keys())
        push_state = np.zeros(self.num_push_sites)
        pull_state = np.zeros(self.num_pull_sites)

        times = []
        push_counts = {mtoc_id: [] for mtoc_id in mtoc_ids}
        pull_counts = {mtoc_id: [] for mtoc_id in mtoc_ids}

        delta_idx = 0
        for trace_file in trace_files:
            changes = np.load(trace_file, allow_pickle=True)
            for push, lattice_site, site_value in changes:
                if push:
                    push_state[int(lattice_site)] = site_value
                else:
                    pull_state[int(lattice_site)] = site_value
                delta_idx += 1
                times.append(delta_idx * self.evolution_time)
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
            ylim (tuple): if given, (ymin, ymax) limits for the # MTs attached axis
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
        ax.set_ylabel('# MTs attached')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plot_folder_path, 'num_mts_per_mtoc.png')
        plt.savefig(save_path)
        plt.close()
        print(f'# MTs per MTOC plot saved to {save_path}')

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

    def calculate_motor_occupancy(self, mtoc_id, start_time, end_time):

        start_idx = int(np.round(start_time, 4) / self.evolution_time)
        end_idx = int(np.round(end_time, 4) / self.evolution_time)

        spindle_trace_dir = os.path.join(self.dir_path, 'spindle_trace')
        trace_files = sorted(
            glob.glob(os.path.join(spindle_trace_dir, 'spindle_trace_*.npy')),
            key=lambda f: int(os.path.basename(f).split('_')[2])
        )

        pull_state = np.zeros(self.num_pull_sites)
        occupancy = np.zeros(self.num_pull_sites)
        delta_idx = 0

        for trace_file in trace_files:
            changes = np.load(trace_file, allow_pickle=True)
            for push, lattice_site, site_value in changes:
                if delta_idx >= end_idx:
                    break
                if delta_idx >= start_idx:
                    occupancy += (pull_state == mtoc_id).astype(int)
                if not push:
                    pull_state[int(lattice_site)] = site_value
                delta_idx += 1
            if delta_idx >= end_idx:
                break

        num_states = end_idx - start_idx
        if num_states > 0:
            occupancy = occupancy / num_states
        return occupancy

    def calculate_push_occupancy(self, mtoc_id, start_time, end_time):

        start_idx = int(np.round(start_time, 4) / self.evolution_time)
        end_idx = int(np.round(end_time, 4) / self.evolution_time)

        spindle_trace_dir = os.path.join(self.dir_path, 'spindle_trace')
        trace_files = sorted(
            glob.glob(os.path.join(spindle_trace_dir, 'spindle_trace_*.npy')),
            key=lambda f: int(os.path.basename(f).split('_')[2])
        )

        push_state = np.zeros(self.num_push_sites)
        occupancy = np.zeros(self.num_push_sites)
        delta_idx = 0

        for trace_file in trace_files:
            changes = np.load(trace_file, allow_pickle=True)
            for push, lattice_site, site_value in changes:
                if delta_idx >= end_idx:
                    break
                if delta_idx >= start_idx:
                    occupancy += (push_state == mtoc_id).astype(int)
                if push:
                    push_state[int(lattice_site)] = site_value
                delta_idx += 1
            if delta_idx >= end_idx:
                break

        num_states = end_idx - start_idx
        if num_states > 0:
            occupancy = occupancy / num_states
        return occupancy

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

        spindle_trace_dir = os.path.join(self.dir_path, 'spindle_trace')
        trace_files = sorted(
            glob.glob(os.path.join(spindle_trace_dir, 'spindle_trace_*.npy')),
            key=lambda f: int(os.path.basename(f).split('_')[2])
        )

        num_sites = self.num_push_sites if push else self.num_pull_sites
        state = np.zeros(num_sites)
        birth_time = np.full(num_sites, np.nan)
        site_indices = []
        lifetimes = []
        delta_idx = 0

        for trace_file in trace_files:
            changes = np.load(trace_file, allow_pickle=True)
            for is_push, lattice_site, site_value in changes:
                if delta_idx >= end_idx:
                    break
                if is_push == push:
                    lattice_site = int(lattice_site)
                    current_time = delta_idx * self.evolution_time
                    old_value = state[lattice_site]
                    if site_value != 0 and old_value == 0:
                        birth_time[lattice_site] = current_time
                    elif site_value == 0 and old_value == mtoc_id and delta_idx >= start_idx:
                        site_indices.append(lattice_site)
                        lifetimes.append(current_time - birth_time[lattice_site])
                    state[lattice_site] = site_value
                delta_idx += 1
            if delta_idx >= end_idx:
                break

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
        times = sorted(list(self.trajectory.keys())[:-1000])
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
                    if (occupancy_frames or push_occupancy_frames):
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
            if occ_scatter is not None:
                occ_scatter.set_facecolors(_pull_rgba(occupancy_frames[frame]))
            if push_occ_scatter is not None:
                push_occ_scatter.set_facecolors(_push_rgba(push_occupancy_frames[frame]))
            cost, tubulin_use, num_mts, dist, forces = _stats(t)
            info_text.set_text(_format_text(t, cost, tubulin_use, num_mts, dist, forces))
            return scatters + ([occ_scatter] if occ_scatter is not None else []) + ([push_occ_scatter] if push_occ_scatter is not None else []) + [info_text]

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