import sys, os
sys.path.insert(0, '/Users/emrealca/Documents/Penn/flatiron-microtubules/multi-aster')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import multi_aster_spindle as mas

np.set_printoptions(suppress=True, precision=6)

dir_path = '/tmp/verify_disk_animation_dummy'
os.makedirs(dir_path, exist_ok=True)

s = mas.Spindle(
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
    euler_timestep_size=1e-4,
    evolution_time=1e-2,
    save_trajectory=True,
    save=False,
    dir_path=dir_path,
)

# attach a couple of synthetic MTs to disk sites so occupancy has something to show
s.add_microtubules_to_disk(mtoc_id=1, site_indices=np.array([0, 1]), push=True, front=True)
s.add_microtubules_to_disk(mtoc_id=2, site_indices=np.array([0]), push=False, front=False)

# manually build a batch of trajectory frames (bypassing optimize()'s stochastic sampler,
# which we're not testing here) by repeatedly running time_evolution and snapshotting exactly
# what optimize()'s save_trajectory block now saves. animate_mtoc_trajectory drops the most
# recent 1000 raw trajectory keys (a pre-existing, unrelated quirk), so we need > 1000 frames
# for anything to remain after that.
n_frames = 1005
for step in range(n_frames):
    new_mtoc_positions, boundary_violated, _ = s.time_evolution()
    s.mtoc_positions = new_mtoc_positions
    s.time += s.evolution_time

    total_force = {mid: s.cytoplasmic_drag_factor * s.calc_mtoc_velocity(mid) for mid in s.mtoc_positions}
    s.trajectory[s.time] = {
        'mtoc_pos': s.mtoc_positions,
        'cost': 0.0,
        'tubulin_use': s.calculate_tubulin_use(),
        'num_mts': s.calculate_num_mts(),
        'total_force': total_force,
        'disk_center': s.disk_center.copy(),
        'disk_e1': s.disk_e1.copy(),
        'disk_e2': s.disk_e2.copy(),
        'disk_e3': s.disk_e3.copy(),
        'disk_push_state_front': s.disk_push_state_front.copy(),
        'disk_push_state_back': s.disk_push_state_back.copy(),
        'disk_pull_state_front': s.disk_pull_state_front.copy(),
        'disk_pull_state_back': s.disk_pull_state_back.copy(),
    }

print(f"built {len(s.trajectory)} trajectory frames")
print("disk_center drifted:", s.trajectory[list(s.trajectory.keys())[0]]['disk_center'], "->", s.disk_center)

anim = s.animate_mtoc_trajectory(save_path=None, interval=50, stride=1)
# exercise the update closure directly across every frame the function itself would actually
# animate -- it drops the most recent 1000 raw trajectory keys before sorting/striding, so
# replicate that exact filtering here rather than using the full 1005-entry trajectory.
times = sorted(list(s.trajectory.keys())[:-1000])[::1]
disk_patch_verts_by_frame = []
e1_line_data_by_frame = []
site_offsets_by_frame = []

for frame in range(len(times)):
    artists = anim._func(frame)
    disk_patch = artists[len(s.mtoc_positions)]       # scatters (2) then disk_patch
    e1_line = artists[len(s.mtoc_positions) + 1]
    disk_sites_scatter = artists[len(s.mtoc_positions) + 4]

    disk_patch_verts_by_frame.append(np.array(disk_patch.get_verts3d()) if hasattr(disk_patch, 'get_verts3d') else None)
    e1_line_data_by_frame.append(np.array(e1_line.get_data_3d()))
    site_offsets_by_frame.append(np.array(disk_sites_scatter._offsets3d))

print("PASS: update() ran for all frames without raising\n")

e1_start = e1_line_data_by_frame[0]
e1_end = e1_line_data_by_frame[-1]
print("e1 line endpoints, frame 0:\n", e1_start)
print("e1 line endpoints, last frame:\n", e1_end)
assert not np.allclose(e1_start, e1_end), "expected the e1 axis line to move/rotate across frames"
print("PASS: disk characteristic-vector line actually changes across frames\n")

site0_start = site_offsets_by_frame[0][:, 0]
site0_end = site_offsets_by_frame[-1][:, 0]
print("disk site 0 position, frame 0:", site0_start, " last frame:", site0_end)
assert not np.allclose(site0_start, site0_end), "expected disk site lab-frame positions to move as the disk moves"
print("PASS: disk site occupancy markers track the moving disk\n")

# sanity-check the occupancy coloring logic directly against the state arrays
disk_site_state = np.concatenate([s.disk_push_state_front, s.disk_push_state_back,
                                   s.disk_pull_state_front, s.disk_pull_state_back])
n_occupied = int(np.sum(disk_site_state != 0))
print(f"{n_occupied} disk sites occupied out of {len(disk_site_state)} total")
assert n_occupied == 3  # 2 push-front (mtoc 1) + 1 pull-back (mtoc 2)
print("PASS: expected number of occupied disk sites")

print("\nALL TESTS PASSED")
