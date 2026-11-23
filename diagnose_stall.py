#
# Emre Alca
# University of Pennsylvania
# Created on Mon Nov 02 2026
#
# Read-only check of why an experiment's optimizer has stopped accepting proposals.
# Loads the experiment's last saved state (nothing is written) and asks:
#   1. how much does the cost change if the system just evolves for one evolution_time,
#      with no site flipped? (the "drift" every proposal shares) -- and which cost terms
#      does that change come from?
#   2. over a sample of real proposals, how much does each change the cost, and what
#      fraction would pass the Metropolis condition?
# If the drift alone raises the cost by many times the temperature, every proposal is
# rejected for the same reason and the run is frozen in place.
#
# usage: python diagnose_stall.py <experiment_dir> --proposals 200
#

import numpy as np

import spindle as mas

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("experiment_dir", type=str)
parser.add_argument("--proposals", type=int, default=200, help="number of real proposals to evaluate")
args = parser.parse_args()

spindle = mas.retrieve_experiement(experiment_dir=os.path.abspath(args.experiment_dir), save=False)
temperature = spindle.optimization_temperature


def cost_terms(spindle):
    """Each calculate_cost term's contribution, by setting every other prefactor to zero in turn."""
    coefficients = spindle.cost_coefficients
    terms = {}
    try:
        for term in coefficients:
            spindle.cost_coefficients = {key: (value if key == term else 0) for key, value in coefficients.items()}
            terms[term] = spindle.calculate_cost()
    finally:
        spindle.cost_coefficients = coefficients
    return terms


def evolve_without_flip(spindle):
    """Cost terms after one evolution_time with nothing flipped; the spindle is restored afterwards."""
    old_mtoc_positions = spindle.mtoc_positions
    old_disk_pose = (spindle.disk_center, spindle.disk_e1, spindle.disk_e2, spindle.disk_e3)
    new_mtoc_positions, boundary_violated = spindle.time_evolution()
    spindle.mtoc_positions = new_mtoc_positions
    # same order as _evaluate_proposal: the moved plate may now shadow some MTs
    culled = spindle.cull_shadowed_microtubules()
    terms = cost_terms(spindle)
    for culled_is_disk, culled_push, culled_front, culled_site, culled_value in culled:
        spindle._resolve_state_array(culled_is_disk, culled_push, culled_front)[culled_site] = culled_value
    spindle.mtoc_positions = old_mtoc_positions
    spindle.disk_center, spindle.disk_e1, spindle.disk_e2, spindle.disk_e3 = old_disk_pose
    return terms, boundary_violated, len(culled)


print(f'{args.experiment_dir}')
print(f'{spindle.num_accepted_states} accepted / {spindle.num_attempts} attempts at the last save, '
      f't = {spindle.time:.3f} s, {spindle.calculate_num_mts()} MTs, temperature {temperature:g}\n')

before = cost_terms(spindle)
after, boundary_violated, num_culled = evolve_without_flip(spindle)

print('-- 1. one evolution_time with no flip --')
print(f'{"term":32s} {"cost now":>12s} {"after drift":>12s} {"change":>12s} {"change / T":>12s}')
for term in before:
    if spindle.cost_coefficients[term] == 0:
        continue
    change = after[term] - before[term]
    print(f'{term:32s} {before[term]:12.4g} {after[term]:12.4g} {change:12.4g} {change / temperature:12.3g}')
drift = sum(after.values()) - sum(before.values())
print(f'{"total":32s} {sum(before.values()):12.4g} {sum(after.values()):12.4g} {drift:12.4g} {drift / temperature:12.3g}')
print(f'boundary violated: {boundary_violated}, MTs culled by the shadow: {num_culled}')
print(f'Metropolis acceptance of the drift alone: {min(1.0, np.exp(-drift / temperature)) if not boundary_violated else 0.0:.3g}\n')

# -- 2. real proposals, evaluated against the current cost and undone --
current_cost = spindle.calculate_cost()
changes, accepted, violated, nothing_to_propose = [], 0, 0, 0
for _ in range(args.proposals):
    proposal = spindle.sample_spindle_update()
    if proposal is None:
        nothing_to_propose += 1
        continue
    evaluation = spindle._evaluate_proposal(proposal, current_cost)
    spindle._revert_proposal(evaluation)
    changes.append(evaluation['cost'] - current_cost)
    accepted += evaluation['accepted']
changes = np.array(changes)

print(f'-- 2. {args.proposals} real proposals --')
if changes.size:
    print(f'cost change / T:  min {changes.min() / temperature:.3g}   median {np.median(changes) / temperature:.3g}   max {changes.max() / temperature:.3g}')
    print(f'proposals that lower the cost: {np.mean(changes < 0):.1%}')
    print(f'accepted by the Metropolis condition: {accepted} of {changes.size}')
print(f'nothing to propose: {nothing_to_propose}')
