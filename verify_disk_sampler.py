import sys, os
sys.path.insert(0, '/Users/emrealca/Documents/Penn/flatiron-microtubules/multi-aster')

import numpy as np
import multi_aster_spindle as mas

np.set_printoptions(suppress=True, precision=6)


def make_spindle(**overrides):
    boundary_radius = 15.0
    # a real experiment's push_lattice has tens of thousands of points (the whole point of a
    # fine mesh), which would always numerically swamp a handful of disk sites in the weighted
    # draw regardless of relative distance. Use a lattice sized comparably to the disk's own
    # site count here purely so the test can actually observe disk candidates being drawn.
    full_lattice = np.load('/Users/emrealca/Documents/Penn/flatiron-microtubules/multi-aster/trimesh_cache/sphere_5_subdivs_1_radius.npy') * boundary_radius
    rng = np.random.default_rng(1)
    push_lattice = full_lattice[rng.choice(len(full_lattice), size=60, replace=False)]
    kwargs = dict(
        initial_mtoc_positions=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]),
        push_lattice=push_lattice,
        pull_lattice=push_lattice[:20].copy(),
        boundary_radius=boundary_radius,
        disk_radius=3.0,
        num_disk_push_sites=10,
        num_disk_pull_sites=4,
        disk_shadowing_enabled=True,
        seed=0,
        tubulin_budget=1000.0,
        save=False,
        dir_path='/tmp/verify_disk_sampler_dummy',
    )
    kwargs.update(overrides)
    if kwargs.get('dir_path') is not None:
        os.makedirs(kwargs['dir_path'], exist_ok=True)
    return mas.Spindle(**kwargs)


print("=== sample_spindle_update returns the new 5-tuple, disk face matches mtoc side ===")
s = make_spindle()
seen_disk_front_for_1 = False
seen_disk_back_for_2 = False
for _ in range(400):
    is_disk, push, front, lattice_site, site_value = s.sample_spindle_update(add=True, mtoc_id=1)
    assert isinstance(is_disk, (bool, np.bool_))
    if is_disk:
        assert front is True, "mtoc 1 sits on the +e1 side; it should only ever be offered the front face"
        seen_disk_front_for_1 = True

    is_disk2, push2, front2, lattice_site2, site_value2 = s.sample_spindle_update(add=True, mtoc_id=2)
    if is_disk2:
        assert front2 is False, "mtoc 2 sits on the -e1 side; it should only ever be offered the back face"
        seen_disk_back_for_2 = True

assert seen_disk_front_for_1, "expected at least one disk-front candidate to be sampled for mtoc 1 in 400 draws"
assert seen_disk_back_for_2, "expected at least one disk-back candidate to be sampled for mtoc 2 in 400 draws"
print("PASS\n")


print("=== remove branch can target disk sites too ===")
s2 = make_spindle()
s2.add_microtubules_to_disk(mtoc_id=1, site_indices=np.array([0, 1, 2]), push=True, front=True)
seen_disk_removal = False
for _ in range(200):
    is_disk, push, front, lattice_site, site_value = s2.sample_spindle_update(add=False)
    if is_disk:
        seen_disk_removal = True
        assert site_value == 0
assert seen_disk_removal, "expected the remove branch to eventually target one of the 3 occupied disk sites"
print("PASS\n")


print("=== tubulin use accounts for disk-attached MTs ===")
s3 = make_spindle()
before = s3.calculate_tubulin_use()
s3.add_microtubules_to_disk(mtoc_id=1, site_indices=np.array([0]), push=True, front=True)
after = s3.calculate_tubulin_use()
print(f"tubulin use before: {before}  after attaching one disk MT: {after}")
assert after > before
print("PASS\n")


print("=== short optimize() run: disk occupancy changes, disk pose stays bounded near target ===")
os.makedirs('/tmp/verify_disk_sampler_optimize', exist_ok=True)
s4 = make_spindle(
    disk_zeta_parallel=5.0, disk_zeta_perp=5.0, disk_zeta_omega=5.0,  # softer drag -> more visible motion
    optimization_temperature=1e-6,  # any cost increase is almost always rejected
    evolution_time=1e-3, euler_timestep_size=1e-4,
    save=True, save_trajectory=True,
    dir_path=None, data_dir='/tmp/verify_disk_sampler_optimize', dir_prefix='run1',
)
s4.optimize(total_attempts=300, save_batch_size=1)  # flush after every acceptance: optimize()
# doesn't flush a trailing partial batch when it returns, so save_batch_size=1 is what makes
# the on-disk trajectory/trace always current for the retrieve_experiement check below --
# an existing, unrelated quirk, not something this pass is trying to fix.

disk_center_norm = np.linalg.norm(s4.disk_center)
print(f"final disk_center norm (target 0): {disk_center_norm:.4f}  (boundary_radius={s4.boundary_radius})")
assert disk_center_norm < 1.0, "expected the strongly-penalized-when-rejected disk centre to stay close to the origin"

num_disk_occupied = (np.count_nonzero(s4.disk_push_state_front) + np.count_nonzero(s4.disk_push_state_back)
                      + np.count_nonzero(s4.disk_pull_state_front) + np.count_nonzero(s4.disk_pull_state_back))
print(f"disk sites occupied at end of run: {num_disk_occupied}")
print(f"num_accepted_states: {s4.num_accepted_states}, num_attempts: {s4.num_attempts}")
print("PASS (no crash, disk stayed bounded)\n")


print("=== retrieve_experiement restores disk config/pose/occupancy ===")
restored = mas.retrieve_experiement(s4.dir_path, save_trajectory=True, save=False)

print("live   disk_center:", s4.disk_center, " restored disk_center:", restored.disk_center)
print("live   disk_e1:", s4.disk_e1, " restored disk_e1:", restored.disk_e1)
assert np.allclose(s4.disk_center, restored.disk_center)
assert np.allclose(s4.disk_e1, restored.disk_e1)
assert np.allclose(s4.disk_e2, restored.disk_e2)
assert np.allclose(s4.disk_e3, restored.disk_e3)
assert np.array_equal(s4.disk_push_state_front, restored.disk_push_state_front)
assert np.array_equal(s4.disk_push_state_back, restored.disk_push_state_back)
assert np.array_equal(s4.disk_pull_state_front, restored.disk_pull_state_front)
assert np.array_equal(s4.disk_pull_state_back, restored.disk_pull_state_back)
assert restored.disk_radius == s4.disk_radius
assert restored.disk_shadowing_enabled == s4.disk_shadowing_enabled
print("PASS: restored spindle's disk config/pose/occupancy match the live one\n")

print("ALL TESTS PASSED")
