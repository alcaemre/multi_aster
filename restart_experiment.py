#
# Emre Alca
# University of Pennsylvania
# Created on Tue Jun 23 2026
# Last Modified: 2026/06/30 16:48:36
#
import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# exp_dir = f'{ceph}multi_aster/useful/spindle_centring_10.0_lbar_5000_tubulin_0.0005_temp'
# exp_dir = f'{ceph}multi_aster/dt-L/single_aster_positioning_2500_tubulin_0.0005_temp'
# exp_dir = f'{ceph}multi_aster/useful/single_aster_positioning_reduced_tub_cost_1000_tubulin_0.002_temp'
# exp_dir = f'{ceph}multi_aster/dt-L/two_aster_8_spindle_length_2000_tubulin_0.002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/two_aster_14_spindle_length_2000_tubulin_0.002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/sap_force_1000_tubulin_0.002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/exponential_2.0_lbar_10000_tubulin_0.002_temp/'
# exp_dir = f'{ceph}multi_aster/dt-L/uniform_13_dbar_10000_tubulin_0.001_temp'
# exp_dir = f'{ceph}multi_aster/dt-L/uniform_13_dbar_10000_tubulin_0.0001_temp/'
exp_dir = f'{ceph}multi_aster/dt-L/four_aster_13_dbar_10000_tubulin_0.0001_temp/'

spindle = mas.retrieve_experiement(experiment_dir=exp_dir, save_trajectory=True, save=True)

# print(spindle.num_accepted_states)

spindle.optimize(5000000)