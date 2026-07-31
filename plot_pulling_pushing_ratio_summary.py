#
# Emre Alca
# University of Pennsylvania
#

import numpy as np
import matplotlib.pyplot as plt

import multi_aster_spindle as mas


def pulling_pushing_ratio_over_window(spindle, window):
    """Ratio of total pulling MTs to total pushing MTs (summed over MTOCs) at each step within the last `window` seconds."""
    end_time = np.round(spindle.time)
    start_time = end_time - window

    times, push_counts, pull_counts = spindle.calculate_num_mts_per_mtoc_over_time()

    mtoc_ids = sorted(push_counts.keys())
    total_push = np.sum([push_counts[mtoc_id] for mtoc_id in mtoc_ids], axis=0)
    total_pull = np.sum([pull_counts[mtoc_id] for mtoc_id in mtoc_ids], axis=0)

    mask = times >= start_time
    total_push = total_push[mask]
    total_pull = total_pull[mask]

    ratios = np.divide(total_pull, total_push, out=np.full_like(total_pull, np.nan, dtype=float), where=total_push != 0)

    return ratios


def experiment_ratio_stats(experiment_dir, window=100):
    """Tubulin budget, and mean/stdev of pulling/pushing MT ratio over the last `window` seconds of an experiment."""
    spindle = mas.retrieve_experiement(experiment_dir=experiment_dir)
    ratios = pulling_pushing_ratio_over_window(spindle, window)
    return spindle.tubulin_budget, np.nanmean(ratios), np.nanstd(ratios)


def plot_pulling_pushing_ratio_summary(experiment_dirs, window=100, out='pulling_pushing_ratio_summary.png'):
    """Plot mean +/- stdev pulling/pushing MT ratio (last `window` seconds) vs. tubulin budget, across experiments.

    Args:
        experiment_dirs (list[str]): paths to experiment folders.
        window (float): trailing time window in seconds.
        out (str): output plot path.
    """
    tubulin_budgets, means, stdevs = [], [], []
    for experiment_dir in experiment_dirs:
        tubulin_budget, mean, stdev = experiment_ratio_stats(experiment_dir, window=window)
        tubulin_budgets.append(tubulin_budget)
        means.append(mean)
        stdevs.append(stdev)
        print(f'{experiment_dir}: tubulin budget = {tubulin_budget}, pull/push ratio = {mean:.3f} +/- {stdev:.3f}')

    order = np.argsort(tubulin_budgets)
    tubulin_budgets = np.array(tubulin_budgets)[order]
    means = np.array(means)[order]
    stdevs = np.array(stdevs)[order]

    plt.figure(figsize=(6, 4))
    plt.errorbar(tubulin_budgets, means, yerr=stdevs, fmt='o', color='#2a78d6', ecolor='#2a78d6',
                 elinewidth=2, capsize=4, markersize=8)
    plt.xlabel('tubulin budget (µm)')
    plt.ylabel('pulling / pushing MT ratio')
    plt.title(f'mean pulling/pushing MT ratio over final {window:g} s')
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
    plot_pulling_pushing_ratio_summary(experiment_dirs, window=100, out='pulling_pushing_ratio_summary.png')
