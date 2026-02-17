import numpy as np
from ocean_wave_tracing.transform.binning import step_heading_theta, circular_mean_rad
from ocean_wave_tracing.transform.geometry import rays_with_points_in_box

def box_postprocess(
    res,
    box,
    min_steps_in_box=1,
    use_ray_theta=False,
    ray_depth_getter=None,
    depth_min=None,
    depth_max=None,
):
    hits = {}
    mean_dir_in_box = {}

    for label, wt in res["cases"].items():
        ray_depth = None
        if ray_depth_getter is not None:
            ray_depth = ray_depth_getter(wt)

        ray_ids, inside = rays_with_points_in_box(
            wt.ray_x, wt.ray_y,
            xmin=box.xmin, xmax=box.xmax,
            ymin=box.ymin, ymax=box.ymax,
            ray_depth=ray_depth,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        hits[label] = ray_ids

        md = {}
        if ray_ids.size == 0:
            mean_dir_in_box[label] = md
            continue

        if use_ray_theta:
            for rid in ray_ids:
                rid = int(rid)
                mask = inside[rid, :]
                if np.count_nonzero(mask) < min_steps_in_box:
                    continue
                th_inside = wt.ray_theta[rid, mask]
                mu = circular_mean_rad(th_inside)
                if np.isfinite(mu):
                    md[rid] = float(np.rad2deg(mu))
        else:
            th_steps = step_heading_theta(wt.ray_x, wt.ray_y)
            nstep = th_steps.shape[1]
            nt = inside.shape[1]

            for rid in ray_ids:
                rid = int(rid)
                inside_r = inside[rid, :]

                if nt == nstep + 1:
                    mask = inside_r[1:]
                    th_row = th_steps[rid, :]
                elif nt == nstep:
                    mask = inside_r
                    th_row = th_steps[rid, :]
                else:
                    L = min(nt, nstep)
                    mask = inside_r[:L]
                    th_row = th_steps[rid, :L]

                if np.count_nonzero(mask) < min_steps_in_box:
                    continue

                th_inside = th_row[mask]
                mu = circular_mean_rad(th_inside)
                if np.isfinite(mu):
                    md[rid] = float(np.rad2deg(mu))

        mean_dir_in_box[label] = md

    return hits, mean_dir_in_box
