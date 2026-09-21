#
# Emre Alca
# University of Pennsylvania
# Created on Thu Aug 13 2026
# Last Modified: 2026/08/13 17:40:04
#

#
# Emre Alca
# University of Pennsylvania
# Created on Thu Aug 13 2026
# Last Modified: Thu Aug 13 2026 5:03:09 PM
#

import math
from typing import List, Tuple


def sunflower_disk_polar(
    n: int,
    radius: float,
    degrees: bool = False,
) -> List[Tuple[float, float]]:
    """
    Centroids of a Fibonacci (sunflower) tessellation of a disk, in polar coords.

    Each returned point is the centroid of an approximately equal-area cell.

        r_k     = radius * sqrt((k - 0.5) / n)      # equal area per point
        theta_k = k * golden_angle                  # even angular spread

    Parameters
    ----------
    n : int
        Number of points / cells.
    radius : float
        Radius of the disk (must be > 0).
    degrees : bool, optional
        If True, angles are returned in degrees in [0, 360).
        Otherwise radians in [0, 2*pi). Default False.

    Returns
    -------
    list of (r, theta) tuples, ordered from the center outward.
    """
    if n <= 0:
        return []
    if radius <= 0:
        raise ValueError("radius must be positive")

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ~2.399963 rad (137.5 deg)
    two_pi = 2.0 * math.pi

    points = []
    for k in range(1, n + 1):
        r = radius * math.sqrt((k - 0.5) / n)
        theta = (k * golden_angle) % two_pi
        if degrees:
            theta = math.degrees(theta)
        points.append((r, theta))
    return points


if __name__ == "__main__":
    for i, (r, theta) in enumerate(sunflower_disk_polar(12, 1.0), start=1):
        print(f"{i:2d}:  r = {r:.4f}   theta = {theta:.4f} rad")