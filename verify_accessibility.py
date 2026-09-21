import sys, os
sys.path.insert(0, '/Users/emrealca/Documents/Penn/flatiron-microtubules/multi-aster')

import numpy as np
import multi_aster_spindle as mas

np.set_printoptions(suppress=True, precision=6)


def make_spindle(**overrides):
    kwargs = dict(
        initial_mtoc_positions=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]),
        push_lattice=np.array([[10.0, 0.0, 0.0]]),
        pull_lattice=np.array([[0.0, 10.0, 0.0]]),
        boundary_radius=20.0,
        disk_radius=2.0,
        num_disk_push_sites=8,
        num_disk_pull_sites=4,
        disk_zeta_parallel=21.33,
        disk_zeta_perp=32.0,
        disk_zeta_omega=85.33,
        save=False,
        dir_path='/tmp/verify_access_dummy',
    )
    kwargs.update(overrides)
    os.makedirs(kwargs['dir_path'], exist_ok=True)
    return mas.Spindle(**kwargs)


s = make_spindle()
# default disk: center=origin, e1=+z. MTOC 1 at (0,0,10) is on +e1 side, MTOC 2 at (0,0,-10) on -e1 side.

print("=== Face reachability ===")
assert s.is_disk_face_reachable(1, front=True) == True
assert s.is_disk_face_reachable(1, front=False) == False
assert s.is_disk_face_reachable(2, front=True) == False
assert s.is_disk_face_reachable(2, front=False) == True
print("PASS: front/back reachability matches which side each MTOC sits on\n")

print("=== Boundary shadowing ===")
# A boundary site directly behind the disk from MTOC 1's point of view (straight down through
# the disk) should be shadowed. A site off to the side should not be.
behind_disk = np.array([[0.0, 0.0, -20.0]])      # straight through the disk, on the far side
beside_disk = np.array([[20.0, 0.0, 0.0]])        # out to the side, ray never comes near the disk
sites = np.vstack([behind_disk, beside_disk])

accessible = s.boundary_sites_accessible(1, sites)
print("accessible from MTOC 1:", accessible)
assert accessible[0] == False, "site directly behind the disk should be shadowed"
assert accessible[1] == True, "site off to the side should be accessible"
print("PASS: shadow test distinguishes blocked vs. clear sites\n")

# sanity check the raw ray-intersection call directly (single direction, and batched)
hit, t = s.disk_ray_intersection(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -1.0]))
print("single-ray hit:", hit, "t*:", t, " (expect hit at t*=10, disk at z=0)")
assert hit and abs(t - 10.0) < 1e-9

hits, ts = s.disk_ray_intersection(np.array([0.0, 0.0, 10.0]), np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]))
print("batched hits:", hits, "t*:", ts, " (second ray travels parallel to the disk plane, never hits)")
assert hits[0] and not hits[1]
print("PASS: disk_ray_intersection handles single and batched directions\n")

print("ALL TESTS PASSED")
