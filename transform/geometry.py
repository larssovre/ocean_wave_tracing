import numpy as np

def rays_with_points_in_box(
    ray_x, ray_y,
    xmin, xmax, ymin, ymax,
    ray_depth=None, depth_min=None, depth_max=None
):
    inside = (
        (ray_x >= xmin) & (ray_x <= xmax) &
        (ray_y >= ymin) & (ray_y <= ymax)
    )

    if ray_depth is not None and (depth_min is not None or depth_max is not None):
        if ray_depth.shape != inside.shape:
            raise ValueError(
                f"ray_depth shape {ray_depth.shape} must match ray_x/ray_y shape {inside.shape}"
            )

        depth_ok = np.ones_like(inside, dtype=bool)
        if depth_min is not None:
            depth_ok &= (ray_depth >= depth_min)
        if depth_max is not None:
            depth_ok &= (ray_depth <= depth_max)

        inside &= depth_ok

    ray_hit = np.any(inside, axis=1)
    ray_ids = np.flatnonzero(ray_hit)
    return ray_ids, inside

