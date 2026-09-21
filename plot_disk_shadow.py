#
# Emre Alca
# University of Pennsylvania
#
# Interactive 3D view of which boundary lattice sites are shadowed by the
# metaphase plate for a given MTOC position. The disk stays fixed; drag the
# sliders to move the MTOC and watch boundary sites flip between accessible
# (green) and shadowed (red), per boundary_sites_accessible in
# multi_aster_spindle.py (disk-forces.tex section 2).
#
# Run as a plain script from a terminal (needs an interactive matplotlib
# backend, e.g. macosx/TkAgg/QtAgg -- it will not work through a headless
# Agg backend):
#
#     python3 plot_disk_shadow.py
#

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import multi_aster_spindle as mas

TRIMESH_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trimesh_cache')


def load_boundary_lattice(radius, mesh_density=5):
    """
    Loads a cached icosphere lattice from trimesh_cache/ (unit radius, the same
    files two-aster-centring.py etc. use for push_lattice) and scales it to
    `radius`. Returns the full cache, unsampled.

    Args:
        radius (float): boundary radius to scale the (unit-radius) cache to.
        mesh_density (int): which cached subdivision level to load -- see the
            sphere_{mesh_density}_subdivs_1_radius.npy files in trimesh_cache/.

    Returns:
        np.ndarray: shape (K, 3) boundary lattice points.
    """
    cache_path = os.path.join(TRIMESH_CACHE_DIR, f'sphere_{mesh_density}_subdivs_1_radius.npy')
    return np.load(cache_path) * radius


def disk_ring_points(spindle, n=64):
    """Points around the disk's rim, for drawing it as a filled patch."""
    theta = np.linspace(0, 2 * np.pi, n)
    return spindle.disk_center + spindle.disk_radius * (
        np.outer(np.cos(theta), spindle.disk_e2) + np.outer(np.sin(theta), spindle.disk_e3)
    )


def build_spindle(boundary_radius, disk_radius, dir_path, mesh_density=5):
    boundary_sites = load_boundary_lattice(boundary_radius, mesh_density=mesh_density)
    spindle = mas.Spindle(
        initial_mtoc_positions=np.array([[0.0, 0.0, boundary_radius * 0.6]]),
        push_lattice=boundary_sites,
        pull_lattice=boundary_sites[:1].copy(),  # unused by this viewer, just needs to exist
        boundary_radius=boundary_radius,
        disk_radius=disk_radius,
        num_disk_push_sites=1,
        num_disk_pull_sites=1,
        save=False,
        dir_path=dir_path,
    )
    return spindle, boundary_sites


def main():
    boundary_radius = 15.0
    disk_radius = 3.0
    mtoc_id = 1

    dir_path = '/tmp/plot_disk_shadow_dummy'
    os.makedirs(dir_path, exist_ok=True)
    spindle, boundary_sites = build_spindle(boundary_radius, disk_radius, dir_path=dir_path)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    accessible = spindle.boundary_sites_accessible(mtoc_id, boundary_sites)
    colors = np.where(accessible, 'tab:green', 'tab:red')
    scat = ax.scatter(boundary_sites[:, 0], boundary_sites[:, 1], boundary_sites[:, 2],
                       c=colors, s=1, alpha=0.15, depthshade=False)

    disk_ring = disk_ring_points(spindle)
    disk_patch = Poly3DCollection([disk_ring], facecolor='tab:blue', alpha=0.35, edgecolor='navy')
    ax.add_collection3d(disk_patch)

    normal_ends = np.array([spindle.disk_center, spindle.disk_center + disk_radius * spindle.disk_e1])
    ax.plot(*normal_ends.T, color='navy', linewidth=2)  # marks the +e1 ("front") side

    mtoc_pos = spindle.mtoc_positions[mtoc_id]
    mtoc_scat = ax.scatter([mtoc_pos[0]], [mtoc_pos[1]], [mtoc_pos[2]],
                           c='black', marker='*', s=200, depthshade=False)

    ax.set_xlim(-boundary_radius, boundary_radius)
    ax.set_ylim(-boundary_radius, boundary_radius)
    ax.set_zlim(-boundary_radius, boundary_radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    title = ax.set_title('')

    def update_title():
        title.set_text(f'{int(np.sum(accessible))}/{len(boundary_sites)} boundary sites accessible from MTOC'
                        f' (navy line marks the disk\'s +e1 / front side)')

    update_title()

    ax_x = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_y = plt.axes([0.15, 0.07, 0.7, 0.03])
    ax_z = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider_x = Slider(ax_x, 'MTOC x', -boundary_radius, boundary_radius, valinit=mtoc_pos[0])
    slider_y = Slider(ax_y, 'MTOC y', -boundary_radius, boundary_radius, valinit=mtoc_pos[1])
    slider_z = Slider(ax_z, 'MTOC z', -boundary_radius, boundary_radius, valinit=mtoc_pos[2])

    def on_slider_change(_):
        nonlocal accessible

        new_pos = np.array([slider_x.val, slider_y.val, slider_z.val])
        spindle.mtoc_positions[mtoc_id] = new_pos

        accessible = spindle.boundary_sites_accessible(mtoc_id, boundary_sites)
        scat.set_color(np.where(accessible, 'tab:green', 'tab:red'))
        mtoc_scat._offsets3d = ([new_pos[0]], [new_pos[1]], [new_pos[2]])

        update_title()
        fig.canvas.draw_idle()

    for slider in (slider_x, slider_y, slider_z):
        slider.on_changed(on_slider_change)

    plt.show()


if __name__ == '__main__':
    main()
