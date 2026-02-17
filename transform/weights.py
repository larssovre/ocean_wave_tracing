import numpy as np

def spectrum_mass(E: np.ndarray, df: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    return np.asarray(E, dtype=float) * df[:, None] * dtheta[None, :]

def mass_to_prob(mass: np.ndarray):
    eps = float(np.sum(mass))
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("Total mass is non-finite or non-positive.")
    P = mass / eps
    P_flat = P.ravel()
    s = float(P_flat.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("Probability sum is non-finite or non-positive.")
    P_flat = P_flat / s
    return P, P_flat

def sample_bins(P_flat: np.ndarray, shape_2d: tuple[int, int], n_samples: int, seed: int):
    rng = np.random.default_rng(int(seed))
    idx_flat = rng.choice(P_flat.size, size=int(n_samples), p=P_flat)
    i_f, j_dir = np.unravel_index(idx_flat, shape_2d)
    return np.asarray(i_f, dtype=int), np.asarray(j_dir, dtype=int)
