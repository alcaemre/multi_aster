#
# Emre Alca
# University of Pennsylvania
#

import numpy as np
import matplotlib.pyplot as plt

import multi_aster_spindle as mas


def spindle_lengths_over_window(spindle, window):
    """Mean pairwise MTOC separation at each saved trajectory timepoint within the last `window` seconds."""
    end_time = np.round(spindle.time)
    start_time = end_time - window

    mtoc_ids = np.array(list(spindle.mtoc_positions.keys()))
    i, j = np.triu_indices(len(mtoc_ids), k=1)
    pairs = np.column_stack([mtoc_ids[i], mtoc_ids[j]])

    lengths = []
    for t, timepoint_data in spindle.trajectory.items():
        if t < start_time:
            continue
        mtoc_pos = timepoint_data['mtoc_pos']
        pair_dists = [mas.normalize_vecs(mtoc_pos[a] - mtoc_pos[b])[1] for a, b in pairs]
        lengths.append(np.mean(pair_dists))

    return np.array(lengths)


def experiment_spindle_length_stats(experiment_dir, window=100):
    """Tubulin budget, and mean/stdev of spindle length over the last `window` seconds of an experiment."""
    spindle = mas.retrieve_experiement(experiment_dir=experiment_dir, save_trajectory=True)
    lengths = spindle_lengths_over_window(spindle, window)
    return spindle.tubulin_budget, lengths.mean(), lengths.std()


def plot_spindle_length_summary(experiment_dirs, window=100, out='spindle_length_summary.png'):
    """Plot mean +/- stdev spindle length (last `window` seconds) vs. tubulin budget, across experiments.

    Args:
        experiment_dirs (list[str]): paths to experiment folders.
        window (float): trailing time window in seconds.
        out (str): output plot path.
    """
    tubulin_budgets, means, stdevs = [], [], []
    for experiment_dir in experiment_dirs:
        tubulin_budget, mean, stdev = experiment_spindle_length_stats(experiment_dir, window=window)
        tubulin_budgets.append(tubulin_budget)
        means.append(mean)
        stdevs.append(stdev)
        print(f'{experiment_dir}: tubulin budget = {tubulin_budget}, spindle length = {mean:.3f} +/- {stdev:.3f} um')

    order = np.argsort(tubulin_budgets)
    tubulin_budgets = np.array(tubulin_budgets)[order]
    means = np.array(means)[order]
    stdevs = np.array(stdevs)[order]

    plt.figure(figsize=(6, 4))
    plt.errorbar(tubulin_budgets, means, yerr=stdevs, fmt='o', color='#2a78d6', ecolor='#2a78d6',
                 elinewidth=2, capsize=4, markersize=8)
    plt.xlabel('tubulin budget (µm)')
    plt.ylabel('spindle length (µm)')
    plt.title(f'mean spindle length over final {window:g} s')
    plt.tight_layout()
    plt.savefig(out)
    print(f'plot saved to {out}')


if __name__ == '__main__':
    ceph = '/mnt/home/ealca/ceph/'
    home = '/mnt/home/ealca/'

    experiment_dirs = [
        f'{ceph}multi_aster/for-spindle-group-meeting/2_aster_both_1000_tubulin_0.005_temperature',
        f'{ceph}multi_aster/for-spindle-group-meeting/2_aster_both_2000_tubulin_0.001_temperature',  
        f'{ceph}multi_aster/for-spindle-group-meeting/2_aster_both_4000_tubulin_0.001_temperature', 
        f'{ceph}multi_aster/for-spindle-group-meeting/2_aster_2.0_lbar_both_6000_tubulin_0.001_temperature',
        f'{ceph}multi_aster/for-spindle-group-meeting/2_aster_2.0_lbar_both_8000_tubulin_0.001_temperature', 
        f'{ceph}multi_aster/for-spindle-group-meeting/2_aster_both_10000_tubulin_0.0005_temperature',   
    ]
    plot_spindle_length_summary(experiment_dirs, window=100, out='spindle_length_summary.png')
