#
# Emre Alca
# University of Pennsylvania
# Created on Fri Jun 12 2026
# Last Modified: 2026/06/18 14:18:01
#

import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import argparse
import os

ceph = '/mnt/home/ealca/ceph/'
home = '/mnt/home/ealca/'

# exp_dir = '/mnt/home/ealca/ceph/multi_aster/dt-L/2_aster_test_pushing_and_pulling_10000_tubulin_0.001_temp/'
exp_dir = f'{ceph}multi_aster/dt-L/elong_further_start_2.0_lbar_5000_tubulin_0.0005_temp'
# exp_dir = f'{ceph}multi_aster/dt-L/2_aster_test_pushing_and_pulling_100_tubulin_0.005_temp/'

if __name__ == "__main__":

    spindle = mas.retrieve_experiement(experiment_dir=exp_dir, save_trajectory=True)
    # spindle.optimize(1000000)

    # # -- plot norm of each aster's position vs time --

    spindle.plot_cost()

    spindle.plot_aster_separation()

    ani_path = os.path.join(spindle.plot_folder_path, f'ani.mp4')
    spindle.animate_mtoc_trajectory(save_path=ani_path, stride=1000)

