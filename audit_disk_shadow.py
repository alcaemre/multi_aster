#
# Emre Alca
# University of Pennsylvania
#
# Audits whether the metaphase plate's shadow actually did anything over a
# finished experiment, by replaying the recorded disk pose against the boundary
# lattice (boundary_sites_accessible in multi_aster_spindle.py).
#
# The point of the audit is that a *correct* shadow can still be invisible. The
# occluded solid angle collapses as the plate turns edge-on to an MTOC: with the
# plate's normal e1 perpendicular to the MTOC axis, every ray from that MTOC runs
# parallel to the plate's plane and nothing at all is blocked. So before
# concluding the shadow is broken, check what pose the plate was actually in.
#
# Usage:
#
#     python3 audit_disk_shadow.py /path/to/experiment_dir
#

import argparse
import os
import pickle

import numpy as np

import multi_aster_spindle as mas


def shadowed_counts(spindle, pose, mtoc_ids):
    """
    Number of boundary push sites each MTOC cannot see, with the disk held at
    `pose` -- a (center, e1, e2, e3) tuple from one trajectory timepoint.

    Restores the spindle's live pose before returning, so the caller can keep
    using the object.
    """
    saved = (spindle.disk_center, spindle.disk_e1, spindle.disk_e2, spindle.disk_e3)
    spindle.disk_center, spindle.disk_e1, spindle.disk_e2, spindle.disk_e3 = pose
    try:
        return {
            mtoc_id: int(np.sum(~spindle.boundary_sites_accessible(mtoc_id, spindle.push_lattice)))
            for mtoc_id in mtoc_ids
        }
    finally:
        spindle.disk_center, spindle.disk_e1, spindle.disk_e2, spindle.disk_e3 = saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', help="directory holding spindle.pkl and trajectory.pkl")
    parser.add_argument('--num_samples', type=int, default=20,
                        help="how many timepoints to sample across the trajectory")
    args = parser.parse_args()

    spindle = mas.retrieve_experiement(args.experiment_dir, save_trajectory=False, save=False)

    mtoc_ids = sorted(spindle.mtoc_positions.keys())
    num_sites = len(spindle.push_lattice)

    times = sorted(spindle.trajectory.keys())
    if not times:
        print('trajectory is empty -- nothing to audit')
        return

    if 'disk_e1' not in spindle.trajectory[times[0]]:
        print('this trajectory predates disk pose tracking -- cannot audit the shadow over time')
        return

    print(f'shadowing enabled in this experiment: {spindle.disk_shadowing_enabled}')
    print(f'{num_sites} boundary push sites, MTOCs {mtoc_ids}, {len(times)} recorded timepoints')
    print()

    # the MTOC axis is what the plate's normal has to stay aligned with for the
    # shadow to occlude anything; for one MTOC, use its own direction from the disk
    if len(mtoc_ids) >= 2:
        axis = mas.normalize_vecs(spindle.mtoc_positions[mtoc_ids[0]]
                                  - spindle.mtoc_positions[mtoc_ids[1]])[0]
    else:
        axis = mas.normalize_vecs(spindle.mtoc_positions[mtoc_ids[0]])[0]

    header = f'{"time (s)":>12}  {"angle(e1, axis)":>16}  ' + '  '.join(f'shadowed[mtoc {i}]' for i in mtoc_ids)
    print(header)
    print('-' * len(header))

    stride = max(1, len(times) // args.num_samples)
    for t in times[::stride]:
        data = spindle.trajectory[t]
        pose = (data['disk_center'], data['disk_e1'], data['disk_e2'], data['disk_e3'])
        angle = np.rad2deg(np.arccos(min(1.0, abs(float(data['disk_e1'] @ axis)))))
        counts = shadowed_counts(spindle, pose, mtoc_ids)
        cells = '  '.join(f'{counts[i]:17d}' for i in mtoc_ids)
        print(f'{t:12.2f}  {angle:15.1f}o  {cells}')

    print()
    print('angle 0 deg  -> plate broadside to the MTOCs, shadow at its largest')
    print('angle 90 deg -> plate edge-on, shadow is exactly zero sites (nothing is "behind" it)')

    # how many MTs currently sit in a place the disk should have blocked
    print()
    print('=== current state: MTs occupying sites their own MTOC cannot see ===')
    for label, state, lattice in (('push', spindle.push_state, spindle.push_lattice),
                                  ('pull', spindle.pull_state, spindle.pull_lattice)):
        occupied = np.where(state != 0)[0]
        for mtoc_id in mtoc_ids:
            mine = occupied[state[occupied] == mtoc_id]
            if mine.size == 0:
                continue
            blocked = int(np.sum(~spindle.boundary_sites_accessible(mtoc_id, lattice[mine])))
            print(f'  {label} / mtoc {mtoc_id}: {blocked} of {mine.size} attached MTs are in shadow')


if __name__ == '__main__':
    main()
