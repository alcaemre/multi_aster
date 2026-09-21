import sys
sys.path.insert(0, '/Users/emrealca/Documents/Penn/flatiron-microtubules/multi-aster')

import numpy as np
import multi_aster_spindle as mas

np.set_printoptions(suppress=True, precision=6)


def make_spindle(**overrides):
    kwargs = dict(
        initial_mtoc_positions=np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]),
        push_lattice=np.array([[10.0, 0.0, 0.0]]),  # dummy 1-site boundary lattices, unused here
        pull_lattice=np.array([[0.0, 10.0, 0.0]]),
        boundary_radius=20.0,
        disk_radius=2.0,
        num_disk_push_sites=8,
        num_disk_pull_sites=4,
        disk_zeta_parallel=21.33,
        disk_zeta_perp=32.0,
        disk_zeta_omega=85.33,
        save=False,
        dir_path='/tmp/verify_disk_dummy',
    )
    kwargs.update(overrides)
    import os
    os.makedirs(kwargs['dir_path'], exist_ok=True)
    return mas.Spindle(**kwargs)


def check_orthonormal(s, label):
    e1, e2, e3 = s.disk_e1, s.disk_e2, s.disk_e3
    print(f"[{label}] |e1|={np.linalg.norm(e1):.6f} |e2|={np.linalg.norm(e2):.6f} |e3|={np.linalg.norm(e3):.6f}"
          f" e1.e2={np.dot(e1,e2):.2e} e1.e3={np.dot(e1,e3):.2e} e2.e3={np.dot(e2,e3):.2e}")


print("=== Test 1: two diametrically opposite pushing MTs on the same face, equal radius ===")
s = make_spindle(initial_mtoc_positions=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]))
# disk default: center at origin, e1=+z, e2=+x, e3=+y (front face normal = +z)
# force two sites to be EXACTLY antipodal (the discrete sunflower tessellation doesn't
# guarantee this for arbitrary n), so the symmetry argument is exact, not approximate.
i, j = 0, 1
s.disk_push_local[i] = np.array([1.0, 0.0])
s.disk_push_local[j] = np.array([-1.0, 0.0])
print("chosen antipodal pair:", i, j, s.disk_push_local[i], s.disk_push_local[j])

# attach MTOC 1 pushing against site i (front), and also against site j (front) -- both MTs from the same MTOC
# growing from z=+10 down to the disk at z=0, hitting front face (+z normal) is correct since mtoc is above disk
s.add_microtubules_to_disk(mtoc_id=1, site_indices=np.array([i, j]), push=True, front=True)

U, omega = s.calculate_disk_velocity_and_omega()
print("U =", U, " (expect ~along -z or 0 in-plane, but nonzero z since pushed from above)")
print("omega =", omega, " (expect ~0, symmetric antipodal pushing MTs cancel torque)")
assert np.linalg.norm(omega) < 1e-8, "expected zero net torque from symmetric antipodal pushing MTs"
print("PASS: zero net torque\n")


print("=== Test 2: single off-center pushing MT produces nonzero omega ===")
s2 = make_spindle(initial_mtoc_positions=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]))
local2 = s2.disk_push_local
site0 = 0
a, b = local2[site0]
print("site 0 local coords (a,b) =", a, b, " r =", np.hypot(a, b))
s2.add_microtubules_to_disk(mtoc_id=1, site_indices=np.array([site0]), push=True, front=True)
U2, omega2 = s2.calculate_disk_velocity_and_omega()
print("U2 =", U2)
print("omega2 =", omega2)
assert np.linalg.norm(omega2) > 1e-8, "expected nonzero torque from a single off-center pushing MT"
print("PASS: nonzero torque\n")


print("=== Test 3: pulling vs pushing MT at the same site direction give opposite-signed translation ===")
s3 = make_spindle(initial_mtoc_positions=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]))
site0 = 0
s3.add_microtubules_to_disk(mtoc_id=1, site_indices=np.array([site0]), push=True, front=True)
U_push, _ = s3.calculate_disk_velocity_and_omega()
s3.remove_microtubules_from_disk(site_indices=np.array([site0]), push=True, front=True)

s3b = make_spindle(initial_mtoc_positions=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]))
pull_site0 = 0
s3b.add_microtubules_to_disk(mtoc_id=1, site_indices=np.array([pull_site0]), push=False, front=True)
U_pull, _ = s3b.calculate_disk_velocity_and_omega()

print("U_push (normal component) =", np.dot(U_push, s3.disk_e1))
print("U_pull (normal component) =", np.dot(U_pull, s3b.disk_e1))
assert np.dot(U_push, s3.disk_e1) * np.dot(U_pull, s3b.disk_e1) < 0, "expected opposite-signed normal velocity"
print("PASS: opposite signs\n")


print("=== Test 4: time_evolution moves the disk and keeps the frame orthonormal ===")
s4 = make_spindle(initial_mtoc_positions=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]),
                   euler_timestep_size=1e-4, evolution_time=1e-2)
site0 = 0
s4.add_microtubules_to_disk(mtoc_id=1, site_indices=np.array([site0]), push=True, front=True)
check_orthonormal(s4, "before")
center_before = s4.disk_center.copy()
s4.time_evolution()
check_orthonormal(s4, "after 100 substeps")
print("disk_center before:", center_before, " after:", s4.disk_center)
assert not np.allclose(center_before, s4.disk_center), "expected the disk center to move"
print("PASS: disk moved and frame stayed orthonormal\n")

print("ALL TESTS PASSED")
