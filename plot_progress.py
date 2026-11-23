#
# Emre Alca
# University of Pennsylvania
# Created on Wed Oct 28 2026
#
# Read-only progress check on an experiment, safe to run while its simulation is still
# going. Loads the experiment from disk (retrieve_experiement with save=False writes
# nothing) and makes the standard plot set (spindle.make_plots) into <experiment>/plots/,
# overwriting the previous set -- the same files the run's own plotting writes.
#
# usage: python plot_progress.py <experiment_dir>
#        python plot_progress.py <experiment_dir> --window 100 --force both    (optional settings, shown at their defaults)
#

import spindle as mas

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("experiment_dir", type=str)
parser.add_argument("--window", type=float, default=100, help="s, window over which occupancy and lifetimes are averaged")
parser.add_argument("--force", type=str, default='both', choices=['push', 'pull', 'both'])
args = parser.parse_args()

exp_dir = os.path.abspath(args.experiment_dir)
if not os.path.isfile(os.path.join(exp_dir, 'spindle.pkl')):
    parser.error(f'{exp_dir} is not an experiment directory (no spindle.pkl). Pass the full path, '
                 f'e.g. /mnt/home/ealca/ceph/multi-aster-parallel/<experiment>')

print(f'checking progress of {exp_dir}')

# save=False: the constructor only sets paths, and nothing below calls optimize(), so the
# only files written are the plots
spindle = mas.retrieve_experiement(experiment_dir=exp_dir, save=False)

mas.make_plots(spindle, force=args.force, window=args.window)
