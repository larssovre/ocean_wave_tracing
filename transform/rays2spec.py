import numpy as np
import pandas as pd
import re
from .binning import circular_nearest_bin, math_deg_to_ocean_deg


def rays2spec(spec2rays_result: dict):
    spec = spec2rays_result["spec"]
    cases = spec2rays_result["cases"]
    hits = spec2rays_result["hits"]
    mean_dir_in_box = spec2rays_result["mean_dir_in_box"]

    ndir = spec.dirs_sorted_rad.size
    nf = spec.freq_hz.size
    twopi = 2.0 * np.pi

    W = np.zeros((nf, ndir), dtype=float)
    C = np.zeros((nf, ndir), dtype=int)

    for label, wt in cases.items():
        m = re.search(r"sample_f=([0-9.]+)Hz", label)
        if m is None:
            continue
        f_lab = float(m.group(1))
        fi = int(np.argmin(np.abs(spec.freq_hz - f_lab)))

        ray_ids_hit = hits.get(label, np.array([], dtype=int))
        if ray_ids_hit.size == 0:
            continue

        mdict = mean_dir_in_box.get(label, {})

        for rid in ray_ids_hit:
            mu_deg = mdict.get(int(rid), np.nan)
            if not np.isfinite(mu_deg):
                continue

            mu_rad = np.deg2rad(mu_deg) % twopi
            j_sorted = circular_nearest_bin(mu_rad, spec.dirs_sorted_rad)

            W[fi, j_sorted] += 1.0
            C[fi, j_sorted] += 1

    W_sum = float(W.sum())
    W_norm = W / W_sum if W_sum > 0 else W * np.nan

    df_long = pd.DataFrame({
        "freq_hz": np.repeat(spec.freq_hz, ndir),
        "j_sorted": np.tile(np.arange(ndir), nf),
        "dir_center_rad_sorted": np.tile(spec.dirs_sorted_rad, nf),
        "dir_center_deg_sorted": np.tile((np.degrees(spec.dirs_sorted_rad) % 360.0), nf),
        "count": C.ravel(),
        "weight": W.ravel(),
        "weight_norm": W_norm.ravel(),
    })

    df_long["j_orig"] = spec.inv_order[df_long["j_sorted"].to_numpy()]
    df_long["dir_center_rad_orig"] = spec.dirs_math_unsorted_rad[df_long["j_orig"].to_numpy()]
    df_long["dir_center_deg_orig"] = (np.degrees(df_long["dir_center_rad_orig"]) % 360.0)
    df_long["dir_deg_ocean"] = math_deg_to_ocean_deg(df_long["dir_center_deg_orig"].to_numpy())

    spec2d = df_long.pivot_table(
        index="freq_hz",
        columns="dir_deg_ocean",
        values="weight_norm",
        aggfunc="sum",
        fill_value=0.0
    ).sort_index(axis=0).sort_index(axis=1)

    return df_long, spec2d


