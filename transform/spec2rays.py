import numpy as np
import pandas as pd

from ocean_wave_tracing.ocean_wave_tracing import Wave_tracing

from ocean_wave_tracing.transform.spectrum_types import Spectrum2D
from ocean_wave_tracing.transform.binning import (
    to_math_dirs_rad,
    sort_dirs_and_E,
    linear_bin_widths,
    cyclic_bin_widths_centers,
    spread_dirs_within_bins,
)
from ocean_wave_tracing.transform.weights import spectrum_mass, mass_to_prob, sample_bins


def _df_to_arrays(df_spec: pd.DataFrame, freq_col: str, dir_col: str, E_col: str):
    freq = np.sort(df_spec[freq_col].unique().astype(float))
    dirs_rad = np.sort(df_spec[dir_col].unique().astype(float))
    nf = freq.size
    nd = dirs_rad.size
    E = np.zeros((nf, nd), dtype=float)
    fi = {float(f): i for i, f in enumerate(freq)}
    dj = {float(d): j for j, d in enumerate(dirs_rad)}
    for row in df_spec[[freq_col, dir_col, E_col]].itertuples(index=False):
        E[fi[float(row[0])], dj[float(row[1])]] = float(row[2])
    return freq, dirs_rad, E


def _build_spectrum(freq: np.ndarray, dirs_rad: np.ndarray, E: np.ndarray):
    dirs_math_unsorted = to_math_dirs_rad(dirs_rad)
    dirs_sorted, E_sorted, order, inv_order = sort_dirs_and_E(dirs_math_unsorted, E)
    spec = Spectrum2D(
        freq_hz=np.asarray(freq, dtype=float),
        dirs_sorted_rad=np.asarray(dirs_sorted, dtype=float),
        E_sorted=np.asarray(E_sorted, dtype=float),
        dirs_math_unsorted_rad=np.asarray(dirs_math_unsorted, dtype=float),
        order=np.asarray(order, dtype=int),
        inv_order=np.asarray(inv_order, dtype=int),
    )
    df = linear_bin_widths(spec.freq_hz)
    dtheta = linear_bin_widths(spec.dirs_sorted_rad)
    dtheta_spread = cyclic_bin_widths_centers(spec.dirs_sorted_rad, 2.0 * np.pi)
    return spec, df, dtheta, dtheta_spread


def _compute_proxy_nt(wave_period_s: float, ray_trace_duration_s: float):
    T0 = float(ray_trace_duration_s)
    if wave_period_s < 5:
        proxy_nt = int(T0 / 1.0)
    elif wave_period_s < 7.5:
        proxy_nt = int(T0 / 0.5)
    elif wave_period_s < 10:
        proxy_nt = int(T0 / 0.5)
    elif wave_period_s < 13:
        proxy_nt = int(T0 / 0.4)
    elif wave_period_s < 16:
        proxy_nt = int(T0 / 0.1)
    else:
        proxy_nt = int(T0 / 0.1)
    #proxy_nt = int(proxy_nt / 4)
    return max(1, proxy_nt)


def spec2rays(
    df_spec: pd.DataFrame,
    U: np.ndarray,
    V: np.ndarray,
    d: np.ndarray,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    domain_X0: float,
    domain_XN: float,
    domain_Y0: float,
    domain_YN: float,
    nt: int,
    T: float,
    launch_geometry,
    n_ray_samples: int = 500,
    random_seed: int = 123,
    freq_col: str = "freq_hz",
    dir_col: str = "dir_rad",
    E_col: str = "E",
    ray_trace_duration_s: float = 2000.0,
):
    freq, dirs_rad, E = _df_to_arrays(df_spec, freq_col, dir_col, E_col)
    spec, df, dtheta, dtheta_spread = _build_spectrum(freq, dirs_rad, E)

    mass = spectrum_mass(spec.E_sorted, df, dtheta)
    _, P_flat = mass_to_prob(mass)
    i_f, j_dir = sample_bins(P_flat, mass.shape, int(n_ray_samples), int(random_seed))

    cases = {}

    for fi in np.unique(i_f):
        mask = i_f == fi
        j_sel = j_dir[mask]
        if j_sel.size == 0:
            continue

        theta0_rad = spread_dirs_within_bins(spec.dirs_sorted_rad, dtheta_spread, j_sel)
        if theta0_rad.size == 0:
            continue

        wave_period_s = 1.0 / float(spec.freq_hz[fi])
        label = f"sample_f={spec.freq_hz[fi]:.6f}Hz"

        if isinstance(launch_geometry, str):
            incoming_wave_side = launch_geometry
            wt = Wave_tracing(
                U, V,
                nx=int(nx), ny=int(ny),
                nt=int(nt), T=float(T),
                dx=float(dx), dy=float(dy),
                nb_wave_rays=int(theta0_rad.size),
                domain_X0=float(domain_X0), domain_XN=float(domain_XN),
                domain_Y0=float(domain_Y0), domain_YN=float(domain_YN),
                d=d,
            )
            wt.set_initial_condition(
                wave_period=float(wave_period_s),
                theta0=theta0_rad,
                incoming_wave_side=incoming_wave_side,
            )
            wt.solve()

        elif isinstance(launch_geometry, tuple) and len(launch_geometry) == 2:
            launch_x_m, launch_y_m = launch_geometry
            ipx = np.full(theta0_rad.size, float(launch_x_m), dtype=float)
            ipy = np.full(theta0_rad.size, float(launch_y_m), dtype=float)
            proxy_nt = _compute_proxy_nt(wave_period_s, float(ray_trace_duration_s))
            wt = Wave_tracing(
                U, V,
                nx=int(nx), ny=int(ny),
                nt=int(proxy_nt), T=float(ray_trace_duration_s),
                dx=float(dx), dy=float(dy),
                nb_wave_rays=int(theta0_rad.size),
                domain_X0=float(domain_X0), domain_XN=float(domain_XN),
                domain_Y0=float(domain_Y0), domain_YN=float(domain_YN),
                d=d,
            )
            wt.set_initial_condition(
                wave_period=float(wave_period_s),
                theta0=theta0_rad,
                ipx=ipx,
                ipy=ipy,
            )
            wt.solve()

        elif isinstance(launch_geometry, int):
            n_pts = int(launch_geometry)
            x_pts = np.full(n_pts, float(domain_X0), dtype=float)
            y_pts = np.linspace(float(domain_Y0), float(domain_YN), n_pts)
            theta = np.asarray(theta0_rad, dtype=float)
            n_rays_per_point = theta.size
            theta_all = np.tile(theta, n_pts)
            ipx_all = np.repeat(x_pts, n_rays_per_point)
            ipy_all = np.repeat(y_pts, n_rays_per_point)
            proxy_nt = _compute_proxy_nt(wave_period_s, float(ray_trace_duration_s))
            wt = Wave_tracing(
                U, V,
                nx=int(nx), ny=int(ny),
                nt=int(proxy_nt), T=float(ray_trace_duration_s),
                dx=float(dx), dy=float(dy),
                nb_wave_rays=int(theta_all.size),
                domain_X0=float(domain_X0), domain_XN=float(domain_XN),
                domain_Y0=float(domain_Y0), domain_YN=float(domain_YN),
                d=d,
            )
            wt.set_initial_condition(
                wave_period=float(wave_period_s),
                theta0=theta_all,
                ipx=ipx_all,
                ipy=ipy_all,
            )
            wt.solve()

        else:
            raise ValueError("Invalid launch_geometry format.")

        cases[label] = wt

    return {
        "spec": spec,
        "df": df,
        "dtheta": dtheta,
        "dtheta_spread": dtheta_spread,
        "mass": mass,
        "P_flat": P_flat,
        "i_f": i_f,
        "j_dir": j_dir,
        "cases": cases,
        "hits": {},
        "mean_dir_in_box": {},
    }
