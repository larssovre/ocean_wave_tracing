import numpy as np

def to_math_dirs_rad(dirs_rad: np.ndarray) -> np.ndarray:
    twopi = 2.0 * np.pi
    return (np.pi / 2.0 - np.asarray(dirs_rad, dtype=float)) % twopi

def sort_dirs_and_E(dirs_math_unsorted: np.ndarray, E: np.ndarray):
    order = np.argsort(dirs_math_unsorted)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    dirs_sorted = np.asarray(dirs_math_unsorted, dtype=float)[order]
    E_sorted = np.asarray(E, dtype=float)[:, order]
    return dirs_sorted, E_sorted, order, inv_order

def linear_bin_widths(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    dx = np.empty_like(x)
    dx[1:-1] = 0.5 * (x[2:] - x[:-2])
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    return dx

def cyclic_bin_widths_centers(centers: np.ndarray, period: float) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    w = linear_bin_widths(c)
    if c.size < 2:
        return w
    wrap_gap = ((c[0] + period) - c[-1]) % period
    w[0] = 0.5 * ((c[1] - c[0]) + wrap_gap)
    w[-1] = 0.5 * ((c[-1] - c[-2]) + wrap_gap)
    return w

def spread_dirs_within_bins(dirs_centers: np.ndarray, dtheta_spread: np.ndarray, j_sel: np.ndarray) -> np.ndarray:
    twopi = 2.0 * np.pi
    j_sel = np.asarray(j_sel, dtype=int)
    out = np.empty(j_sel.size, dtype=float)
    for j in np.unique(j_sel):
        idx = np.flatnonzero(j_sel == j)
        n = idx.size
        c = float(dirs_centers[j])
        w = float(dtheta_spread[j])
        if n == 1:
            out[idx[0]] = c % twopi
        else:
            left = c - 0.5 * w
            right = c + 0.5 * w
            u = (np.arange(n) + 0.5) / n
            out[idx] = (left + u * (right - left)) % twopi
    return out

def step_heading_theta(ray_x: np.ndarray, ray_y: np.ndarray) -> np.ndarray:
    twopi = 2.0 * np.pi
    dx = ray_x[:, 1:] - ray_x[:, :-1]
    dy = ray_y[:, 1:] - ray_y[:, :-1]
    theta = (np.arctan2(dy, dx) % twopi)
    out = np.full_like(ray_x, np.nan, dtype=float)
    out[:, :-1] = theta
    return out

def circular_mean_rad(theta_rad_1d: np.ndarray) -> float:
    twopi = 2.0 * np.pi
    theta = theta_rad_1d[np.isfinite(theta_rad_1d)]
    if theta.size == 0:
        return np.nan
    s = np.mean(np.sin(theta))
    c = np.mean(np.cos(theta))
    if (s == 0.0) and (c == 0.0):
        return np.nan
    return (np.arctan2(s, c) % twopi)

def circular_nearest_bin(theta_rad: float, centers_rad: np.ndarray) -> int:
    d = np.angle(np.exp(1j * (theta_rad - centers_rad)))
    return int(np.argmin(np.abs(d)))

def math_deg_to_ocean_deg(deg_math: np.ndarray) -> np.ndarray:
    return (450.0 - np.asarray(deg_math, dtype=float)) % 360.0
