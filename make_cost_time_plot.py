#
# Emre Alca
# University of Pennsylvania
# Created on Thu Jun 11 2026
# Last Modified: 2026/06/11 09:12:24
#
 

import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import argparse


# --- set basic numbers from argparse ---
parser = argparse.ArgumentParser()
parser.add_argument("--evolution_time", type=float, default=0.01)
parser.add_argument("--optimization_temperature", type=float, default=0.01)

args = parser.parse_args()
evolution_time = args.evolution_time
optimization_temperature = args.optimization_temperature
timestep_size = evolution_time / 10

print(f'making cost vs time plot for single aster with only pushing force and \n evo_time = {evolution_time} \n temp = {optimization_temperature}')

if timestep_size > 0.001:
    timestep_size = 0.001

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

experiment_dir = f'{ceph}multi_aster/dt-L/time_ev_{evolution_time}_temp_{optimization_temperature}'

spindle = mas.retrieve_experiement(experiment_dir=experiment_dir, save_trajectory=True)
spindle.plot_cost()