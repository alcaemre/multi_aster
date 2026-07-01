#
# Emre Alca
# University of Pennsylvania
# Created on Thu Jun 25 2026
# Last Modified: 2026/06/25 10:27:04
#

import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os

def calculate_pulling_mt_lengths_at_birth(spindle, mtoc_index, start_time, end_time):
    start_time = int(start_time / spindle.evolution_time)
    end_time = int(end_time / spindle.evolution_time)
    
    spindle_trace_path = os.path.join(spindle.dir_path, 'spindle_trace')
    
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
    
    # -- cull this big set of spindle states to include only states between start_time and end_time --
    
    # find the number of states to cull from the start
    start_time_discrepancy = start_time - bottom
    end_time_discrepancy = top - end_time
    
    spindle_states_between_start_end = temp_spindle_states[start_time_discrepancy:]
    if end_time_discrepancy > 0:
        spindle_states_between_start_end = spindle_states_between_start_end[:-end_time_discrepancy]
    
    # -- find birth lengths --
    
    times = np.array(list(spindle.trajectory.keys()))
    
    birth_lengths = []
    
    for i in range(1, len(spindle_states_between_start_end)):
        
        state_difference = (spindle_states_between_start_end[i] - spindle_states_between_start_end[i-1])
        
        if (state_difference == mtoc_index).any(): # states where new MTs impinge on motors
            # print(i)
            birth_index = np.where(state_difference == mtoc_index)[0][0] # known to be only one number
            time_of_birth = times[start_time + i]
            mtoc_pos_at_birth = spindle.trajectory[time_of_birth]['mtoc_pos'][mtoc_index]
            landing_site = spindle.pull_lattice[birth_index]
            mt_length_at_birth = mas.normalize_vecs(mtoc_pos_at_birth - landing_site)[1]
    
            birth_lengths.append(mt_length_at_birth)

    return np.array(birth_lengths)

if __name__ == '__main__':
    ceph = '/mnt/home/ealca/ceph/'
    home = '/mnt/home/ealca/'

    # exp_dir = f'{ceph}multi_aster/useful/spindle_centring_10.0_lbar_5000_tubulin_0.0005_temp'
    # exp_dir = f'{ceph}multi_aster/dt-L/single_aster_positioning_2500_tubulin_0.0005_temp'
    # exp_dir = f'{ceph}multi_aster/useful/single_aster_positioning_reduced_tub_cost_1000_tubulin_0.002_temp'
    exp_dir = f'{ceph}multi_aster/dt-L/two_aster_14_spindle_length_2000_tubulin_0.002_temp/'

    spindle = mas.retrieve_experiement(experiment_dir=exp_dir, save_trajectory=True)

    start_time = 100
    end_time = 500

    # -- find the length of each MT at birth --
    mtoc_index = 1

    # loading files

    pulling_lengths_at_birth = calculate_pulling_mt_lengths_at_birth(spindle, 1, start_time, end_time)

    print(f'mean MT length at birth: {np.mean(pulling_lengths_at_birth)}, std: {np.std(pulling_lengths_at_birth)}')

    num_bins = 40

    fig, ax = plt.subplots()
    ax.set_title(f'MT lengths at birth between t={start_time} and t={end_time}\n Mean: {np.round(np.mean(pulling_lengths_at_birth),3)}, std: {np.round(np.std(pulling_lengths_at_birth),3)}')
    ax.hist(pulling_lengths_at_birth, bins=num_bins)
    ax.set_ylabel('count')
    ax.set_xlabel('MT length (um)')
    mt_birth_lengths_path = os.path.join(spindle.plot_folder_path, 'mt_lengths_at_birth.png')
    plt.savefig(mt_birth_lengths_path)
    plt.close()
    print(f'plot saved to {mt_birth_lengths_path}')

    # -- plot heatmap of motor occupancy --
    pull_lattice = spindle.pull_lattice

    xs = pull_lattice[:,0]
    ys = pull_lattice[:,1]
    zs = pull_lattice[:,2]

    # start_time = 55
    # end_time = 140

    occupancy = spindle.calculate_motor_occupancy(1, start_time, end_time)
    norm = mcolors.Normalize(vmin=occupancy.min(), vmax=occupancy.max())

    cmap = plt.cm.inferno
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f'occupancy proportion from {start_time} to {end_time} seconds')

    ax.scatter(xs, ys, zs, c=occupancy,alpha=0.5, cmap=cmap, norm=norm)
    ax.axis('equal')

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(occupancy)
    fig.colorbar(mappable, ax=ax, shrink=0.5, label="Occupancy")
    motor_occupancy_path = os.path.join(spindle.plot_folder_path, 'motor_occupancy.png')
    plt.savefig(motor_occupancy_path)
    plt.close()
    print(f'plot saved to {motor_occupancy_path}')

