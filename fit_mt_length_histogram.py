#
# Emre Alca
# University of Pennsylvania
# Created on Thu Jun 25 2026
# Last Modified: 2026/06/25 16:56:53
#
import numpy as np

import multi_aster_spindle as mas

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os


def calculate_mt_lengths(spindle, mtoc_index, start_time, end_time):
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

    # # calculate and save the lengths of all MTs
    lengths = []
    for i in range(1, len(spindle_states_between_start_end)):
        pulling_states = spindle.pull_lattice[np.where(spindle_states_between_start_end[i] == mtoc_index)[0]]
        states_lengths = mas.normalize_vecs(pulling_states - spindle.mtoc_positions[mtoc_index])[1]

        lengths.extend(states_lengths)
    
    return lengths


"""
Test whether a dataset is exponentially distributed.

Workflow:
  1. Fit the exponential parameter (lambda) via MLE.
  2. Goodness-of-fit tests: Anderson-Darling, Monte-Carlo KS (Lilliefors-style),
     chi-square on bins.
  3. Diagnostic plots: histogram+PDF, Q-Q plot, log-survival plot.
  4. Coefficient-of-variation check (should be ~1).

Dependencies: numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ----------------------------------------------------------------------
# 1. Parameter estimation
# ----------------------------------------------------------------------
def fit_exponential(data):
    """MLE fit of an exponential with location fixed at 0.

    Returns (scale, lambda_hat) where scale = mean = 1 / lambda_hat.
    """
    _, scale = stats.expon.fit(data, floc=0)
    lambda_hat = 1.0 / scale
    return scale, lambda_hat


# ----------------------------------------------------------------------
# 2. Goodness-of-fit tests
# ----------------------------------------------------------------------
def anderson_darling(data):
    """A-D test. scipy's 'expon' critical values already account for an
    estimated parameter. Reject if statistic > critical value."""
    return stats.anderson(data, dist='expon')


def _ks_statistic_expon(data, scale):
    """One-sample KS distance between the data's ECDF and an exponential
    CDF with the given scale."""
    stat, _ = stats.kstest(data, 'expon', args=(0, scale))
    return stat


def ks_montecarlo(data, n_sim=2000, rng=None):
    """Lilliefors-style KS test for exponentiality with the rate estimated
    from the data.

    The standard KS test assumes a fully specified distribution. When the
    parameter is estimated from the same sample, the KS statistic is
    stochastically smaller, so the ordinary p-value is too conservative.
    We correct for this by simulating the null distribution: draw samples
    from the *fitted* exponential, refit each one, and recompute the KS
    distance. The p-value is the fraction of simulated statistics that
    meet or exceed the observed one.

    Returns (observed_statistic, p_value).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(data)
    scale_hat, _ = fit_exponential(data)
    observed = _ks_statistic_expon(data, scale_hat)

    # The null distribution of the statistic does not depend on the true
    # scale (it's a scale-family pivot), so simulating at scale_hat is fine.
    count = 0
    for _ in range(n_sim):
        sample = rng.exponential(scale=scale_hat, size=n)
        s_scale, _ = fit_exponential(sample)
        s_stat = _ks_statistic_expon(sample, s_scale)
        if s_stat >= observed:
            count += 1

    # +1 in numerator and denominator => never returns exactly 0 (standard
    # Monte-Carlo p-value correction).
    p_value = (count + 1) / (n_sim + 1)
    return observed, p_value


def _merge_small_bins(obs, exp, min_exp=5):
    """Merge adjacent bins until every expected count >= min_exp."""
    obs, exp = list(obs), list(exp)
    i = 0
    while i < len(exp):
        if exp[i] < min_exp and len(exp) > 1:
            j = i + 1 if i + 1 < len(exp) else i - 1
            exp[j] += exp[i]
            obs[j] += obs[i]
            del exp[i]
            del obs[i]
        else:
            i += 1
    return np.array(obs), np.array(exp)


def chi_square_test(data, scale, bins='auto', min_exp=5):
    """Chi-square goodness-of-fit on histogram bins.

    df = (#bins after merging) - 1 - 1, the final -1 for the estimated lambda.
    """
    counts, edges = np.histogram(data, bins=bins)
    cdf = stats.expon.cdf(edges, loc=0, scale=scale)
    expected = np.diff(cdf) * len(data)

    obs_m, exp_m = _merge_small_bins(counts, expected, min_exp=min_exp)

    # Rescale expected to match observed total (mass may fall outside bin range)
    exp_m = exp_m * obs_m.sum() / exp_m.sum()

    if len(obs_m) < 3:
        return np.nan, np.nan, 0  # not enough bins for a meaningful test

    chi2_stat, _ = stats.chisquare(obs_m, exp_m, ddof=1)
    df = len(obs_m) - 1 - 1
    p_chi2 = stats.chi2.sf(chi2_stat, df)
    return chi2_stat, p_chi2, df


# ----------------------------------------------------------------------
# 3. Diagnostic plots
# ----------------------------------------------------------------------
def diagnostic_plots(data, scale, lambda_hat, show=True, savepath=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histogram with fitted density overlaid
    axes[0].hist(data, bins='auto', density=True, alpha=0.6,
                 edgecolor='white', label='data')
    xs_grid = np.linspace(0, np.max(data), 400)
    axes[0].plot(xs_grid, stats.expon.pdf(xs_grid, scale=scale),
                 'r-', lw=2, label=f'fitted exp ($\\lambda$={lambda_hat:.3f})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('density')
    axes[0].set_title('Histogram vs fitted PDF')
    axes[0].legend()

    # Q-Q plot against exponential
    stats.probplot(data, dist=stats.expon, sparams=(0, scale), plot=axes[1])
    axes[1].set_title('Exponential Q–Q plot')

    # Log-survival plot: straight line with slope -lambda if exponential
    xs = np.sort(data)
    sf = 1.0 - (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
    axes[2].plot(xs, np.log(sf), '.', ms=4, label='empirical')
    axes[2].plot(xs, np.log(stats.expon.sf(xs, scale=scale)), 'r-',
                 label=f'fitted (slope $-\\lambda$={-lambda_hat:.3f})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('log S(x)')
    axes[2].set_title('Log-survival plot')
    axes[2].legend()

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------
# 4. Moment check
# ----------------------------------------------------------------------
def coefficient_of_variation(data):
    """CV ~ 1 is a (weak) necessary condition for an exponential."""
    return np.std(data, ddof=1) / np.mean(data)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def analyze_exponential(data, alpha=0.05, show_plots=True, savepath=None,
                        n_sim=2000, seed=None):
    """Run the full exponential goodness-of-fit analysis and print a report."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if np.any(data < 0):
        print("WARNING: data contains negative values; an exponential is "
              "supported only on [0, inf). Results may be meaningless.\n")

    rng = np.random.default_rng(seed)
    scale, lambda_hat = fit_exponential(data)

    print("=" * 60)
    print("EXPONENTIAL DISTRIBUTION GOODNESS-OF-FIT REPORT")
    print("=" * 60)
    print(f"n            = {len(data)}")
    print(f"mean (1/lam) = {scale:.4f}")
    print(f"lambda_hat   = {lambda_hat:.4f}")
    print(f"significance = {alpha}")
    print("-" * 60)

    verdicts = []

    # Anderson-Darling
    ad = anderson_darling(data)
    cv_levels = dict(zip(ad.significance_level, ad.critical_values))
    target_pct = alpha * 100
    closest = min(cv_levels, key=lambda s: abs(s - target_pct))
    ad_crit = cv_levels[closest]
    ad_reject = ad.statistic > ad_crit
    verdicts.append(not ad_reject)
    print("Anderson–Darling")
    print(f"  statistic       = {ad.statistic:.4f}")
    print(f"  crit @ {closest:>4.1f}%   = {ad_crit:.4f}")
    print(f"  -> {'REJECT' if ad_reject else 'fail to reject'} exponentiality")
    print("-" * 60)

    # Monte-Carlo KS (Lilliefors-style)
    ks_stat, ks_p = ks_montecarlo(data, n_sim=n_sim, rng=rng)
    ks_reject = ks_p < alpha
    verdicts.append(not ks_reject)
    print(f"Monte-Carlo KS (estimated param, {n_sim} sims)")
    print(f"  statistic = {ks_stat:.4f}")
    print(f"  p-value   = {ks_p:.4f}")
    print(f"  -> {'REJECT' if ks_reject else 'fail to reject'} exponentiality")
    print("-" * 60)

    # Chi-square
    chi2_stat, chi2_p, df = chi_square_test(data, scale)
    if np.isnan(chi2_stat):
        print("Chi-square: too few bins after merging; skipped")
    else:
        chi2_reject = chi2_p < alpha
        verdicts.append(not chi2_reject)
        print("Chi-square (binned)")
        print(f"  statistic = {chi2_stat:.4f}")
        print(f"  df        = {df}")
        print(f"  p-value   = {chi2_p:.4f}")
        print(f"  -> {'REJECT' if chi2_reject else 'fail to reject'} exponentiality")
    print("-" * 60)

    # Coefficient of variation
    cv = coefficient_of_variation(data)
    print(f"Coefficient of variation = {cv:.3f}  (≈1 expected; weak check)")
    print("=" * 60)

    # Overall summary
    n_pass = sum(verdicts)
    n_tests = len(verdicts)
    print(f"SUMMARY: {n_pass}/{n_tests} tests fail to reject exponentiality.")
    if n_pass == n_tests:
        print("=> Consistent with an exponential distribution.")
    elif n_pass == 0:
        print("=> Strong evidence against an exponential distribution.")
    else:
        print("=> Mixed evidence; inspect the diagnostic plots before deciding.")
    print("Note: failing to reject is not proof. With large n, tests reject on")
    print("trivial deviations, so weigh the plots and effect size too.")
    print("=" * 60)

    if show_plots:
        diagnostic_plots(data, scale, lambda_hat, show=True, savepath=savepath)

    return {
        "scale": scale,
        "lambda_hat": lambda_hat,
        "anderson_darling": {"statistic": ad.statistic, "crit": ad_crit,
                             "reject": ad_reject},
        "ks_montecarlo": {"statistic": ks_stat, "p": ks_p, "reject": ks_reject},
        "chi_square": {"statistic": chi2_stat, "p": chi2_p, "df": df},
        "cv": cv,
    }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Replace this block with your own data ----
    rng = np.random.default_rng(42)
    demo_data = rng.exponential(scale=2.0, size=2000)
    # ------------------------------------------------

    results = analyze_exponential(demo_data, alpha=0.05, show_plots=True,
                                  seed=0)


# ----------------------------------------------------------------------
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

    pulling_mt_lengths = calculate_mt_lengths(1, start_time, end_time)
    pulling_mt_lengths.extend(calculate_mt_lengths(2, start_time, end_time))

    print(len(pulling_mt_lengths))

    results = analyze_exponential(pulling_mt_lengths, alpha=0.5, show_plots=True)