import sys, os
sys.path.insert(0, '/Users/emrealca/Documents/Penn/flatiron-microtubules/multi-aster')

import numpy as np
import multi_aster_spindle as mas

np.set_printoptions(suppress=True, precision=6)


def make_spindle(**overrides):
    boundary_radius = 15.0
    push_lattice = np.load('/Users/emrealca/Documents/Penn/flatiron-microtubules/multi-aster/trimesh_cache/sphere_5_subdivs_1_radius.npy') * boundary_radius
    kwargs = dict(
        initial_mtoc_positions=np.array([[0.0, 0.0, 10.0]]),
        push_lattice=push_lattice,
        pull_lattice=push_lattice[:5].copy(),
        boundary_radius=boundary_radius,
        disk_radius=3.0,     # big enough relative to the MTOC's distance (10) to cast a real shadow
        num_disk_push_sites=8,
        num_disk_pull_sites=4,
        seed=0,
        save=False,
        dir_path='/tmp/verify_sampling_shadow_dummy',
    )
    kwargs.update(overrides)
    os.makedirs(kwargs['dir_path'], exist_ok=True)
    return mas.Spindle(**kwargs)


def undo(spindle, is_disk, push, front, site):
    """Clear whichever site was just proposed, so repeated draws keep finding empty sites."""
    if is_disk:
        state = (spindle.disk_push_state_front if front else spindle.disk_push_state_back) if push \
            else (spindle.disk_pull_state_front if front else spindle.disk_pull_state_back)
    else:
        state = spindle.push_state if push else spindle.pull_state
    state[site] = 0


print("=== disk_shadowing_enabled=False (default): behavior unchanged, can sample shadowed boundary sites ===")
s_off = make_spindle(disk_shadowing_enabled=False)
sampled_shadowed_at_least_once = False
for _ in range(200):
    is_disk, push, front, site, value = s_off.sample_spindle_update(add=True, mtoc_id=1)
    if site == 0 and value == 0:
        continue
    if is_disk:
        undo(s_off, is_disk, push, front, site)
        continue
    site_pos = (s_off.push_lattice[site] if push else s_off.pull_lattice[site])[np.newaxis, :]
    if not s_off.boundary_sites_accessible(1, site_pos)[0]:
        sampled_shadowed_at_least_once = True
        break
    undo(s_off, is_disk, push, front, site)  # undo so repeated calls keep finding empty sites
assert sampled_shadowed_at_least_once, "expected the unfiltered sampler to eventually hit a shadowed boundary site"
print("PASS: default (disabled) sampler is unaffected by the disk\n")


print("=== disk_shadowing_enabled=True: never returns a shadowed boundary site ===")
s_on = make_spindle(disk_shadowing_enabled=True)
for i in range(500):
    is_disk, push, front, site, value = s_on.sample_spindle_update(add=True, mtoc_id=1)
    if site == 0 and value == 0:
        continue
    if not is_disk:
        site_pos = (s_on.push_lattice[site] if push else s_on.pull_lattice[site])[np.newaxis, :]
        accessible = s_on.boundary_sites_accessible(1, site_pos)[0]
        assert accessible, f"sampled a shadowed boundary site at iteration {i}: push={push} site={site}"
    undo(s_on, is_disk, push, front, site)
print("PASS: 500 draws, every returned boundary site was accessible from the mtoc\n")


print("=== every boundary site shadowed: gracefully returns the empty-pool sentinel instead of hanging ===")
# an MTOC sitting essentially on the disk's axis, very close to it, with a disk almost as
# large as the boundary -- should shadow nearly (or all) of the far hemisphere; force the
# extreme case by only offering sites we know are all shadowed, and disable the disk face
# itself as a candidate too (radius 0) so the pool is purely the shadowed boundary.
s_extreme = make_spindle(disk_shadowing_enabled=True, disk_radius=14.9, num_disk_push_sites=0,
                          num_disk_pull_sites=0, initial_mtoc_positions=np.array([[0.0, 0.0, 0.5]]))
# fill every site except ones on the near (accessible) hemisphere to force "all empty == shadowed"
accessible_mask = s_extreme.boundary_sites_accessible(1, s_extreme.push_lattice)
s_extreme.push_state[accessible_mask] = 1  # occupy all accessible sites so only shadowed ones remain empty
n_empty_before = int(np.sum(s_extreme.push_state == 0))
print("empty (all shadowed) sites remaining:", n_empty_before)
result = s_extreme.sample_spindle_update(add=True, mtoc_id=1)
print("result:", result)
assert result == (False, True, None, 0, 0)
print("PASS: returns the empty-pool sentinel instead of looping forever\n")

print("ALL TESTS PASSED")
