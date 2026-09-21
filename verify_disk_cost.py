import sys, os
sys.path.insert(0, '/Users/emrealca/Documents/Penn/flatiron-microtubules/multi-aster')

import numpy as np
import multi_aster_spindle as mas

np.set_printoptions(suppress=True, precision=6)


def make_spindle(**overrides):
    kwargs = dict(
        initial_mtoc_positions=np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]),
        push_lattice=np.array([[10.0, 0.0, 0.0]]),
        pull_lattice=np.array([[0.0, 10.0, 0.0]]),
        boundary_radius=20.0,
        spindle_length=10.0,
        tubulin_budget=1.0,   # keep the (unrelated) material term irrelevant/small for this check
        save=False,
        dir_path='/tmp/verify_disk_cost_dummy',
    )
    kwargs.update(overrides)
    os.makedirs(kwargs['dir_path'], exist_ok=True)
    return mas.Spindle(**kwargs)


print("=== disk exactly at the target (origin, e1 horizontal along +x) ===")
s = make_spindle(disk_center=np.array([0.0, 0.0, 0.0]), disk_normal=np.array([1.0, 0.0, 0.0]),
                  disk_tangent=np.array([0.0, 1.0, 0.0]))
# spindle length is exactly 10 (matches self.spindle_length) so that term is also ~0
s.mtoc_positions[1] = np.array([5.0, 0.0, 0.0])
s.mtoc_positions[2] = np.array([-5.0, 0.0, 0.0])
cost_at_target = s.calculate_cost()
print("cost at exact target:", cost_at_target)
# with no MTs attached, the (unrelated) material term is always exactly
# |1 - 0/tubulin_budget| = 1; the spindle-length and disk terms should contribute nothing
# extra when everything is exactly at its target, so total cost should be exactly 1.0
assert abs(cost_at_target - 1.0) < 1e-9, "expected only the baseline material cost (1.0) when disk/spindle are exactly at target"
print("PASS\n")

print("=== disk off-centre (R_d > 0) increases cost ===")
s2 = make_spindle(disk_center=np.array([0.0, 0.0, 0.0]), disk_normal=np.array([1.0, 0.0, 0.0]),
                   disk_tangent=np.array([0.0, 1.0, 0.0]))
s2.mtoc_positions[1] = np.array([5.0, 0.0, 0.0])
s2.mtoc_positions[2] = np.array([-5.0, 0.0, 0.0])
base_cost = s2.calculate_cost()
s2.disk_center = np.array([3.0, 0.0, 0.0])
moved_cost = s2.calculate_cost()
print(f"base cost: {base_cost}  after moving disk off-centre: {moved_cost}")
assert moved_cost > base_cost
assert abs((moved_cost - base_cost) - 9.0) < 1e-9, "expected the increase to equal disk_position_coefficient * R_d^2 = 1*3^2 = 9"
print("PASS\n")

print("=== disk normal tilted away from horizontal (theta_d != pi/2) increases cost ===")
s3 = make_spindle(disk_center=np.array([0.0, 0.0, 0.0]), disk_normal=np.array([1.0, 0.0, 0.0]),
                   disk_tangent=np.array([0.0, 1.0, 0.0]))
s3.mtoc_positions[1] = np.array([5.0, 0.0, 0.0])
s3.mtoc_positions[2] = np.array([-5.0, 0.0, 0.0])
base_cost = s3.calculate_cost()
# tilt e1 to point straight up (+z): theta_d = 0, so (theta_d - pi/2)^2 = (pi/2)^2
s3.disk_e1 = np.array([0.0, 0.0, 1.0])
s3.disk_e2 = np.array([0.0, 1.0, 0.0])
s3.disk_e3 = np.cross(s3.disk_e1, s3.disk_e2)
tilted_cost = s3.calculate_cost()
expected_increase = (np.pi / 2) ** 2
print(f"base cost: {base_cost}  after tilting e1 to +z: {tilted_cost}  expected increase: {expected_increase}")
assert abs((tilted_cost - base_cost) - expected_increase) < 1e-9
print("PASS\n")

print("=== disk normal rotated in-plane (phi_d != 0) increases cost ===")
s4 = make_spindle(disk_center=np.array([0.0, 0.0, 0.0]), disk_normal=np.array([1.0, 0.0, 0.0]),
                   disk_tangent=np.array([0.0, 1.0, 0.0]))
s4.mtoc_positions[1] = np.array([5.0, 0.0, 0.0])
s4.mtoc_positions[2] = np.array([-5.0, 0.0, 0.0])
base_cost = s4.calculate_cost()
# rotate e1 to +y (still horizontal, theta_d = pi/2, but phi_d = pi/2 now)
s4.disk_e1 = np.array([0.0, 1.0, 0.0])
s4.disk_e2 = np.array([-1.0, 0.0, 0.0])
s4.disk_e3 = np.cross(s4.disk_e1, s4.disk_e2)
rotated_cost = s4.calculate_cost()
expected_increase = (np.pi / 2) ** 2
print(f"base cost: {base_cost}  after rotating e1 to +y: {rotated_cost}  expected increase: {expected_increase}")
assert abs((rotated_cost - base_cost) - expected_increase) < 1e-9
print("PASS\n")

print("ALL TESTS PASSED")
