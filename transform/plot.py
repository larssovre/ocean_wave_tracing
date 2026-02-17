import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_spectrum_2d(spec2d, radius="frequency", r_lim=None, Hs=None, cmap="viridis"):
    df = spec2d.copy()

    if radius == "period":
        df = df.iloc[::-1]
        r_vals = 1.0 / df.index.to_numpy()
        r_label = "Period (s)"
    else:
        r_vals = df.index.to_numpy()
        r_label = "Frequency (Hz)"

    if r_lim is not None:
        r_mask = (r_vals >= r_lim[0]) & (r_vals <= r_lim[1])
        r_vals = r_vals[r_mask]
        df = df.iloc[r_mask]

    dirs = df.columns.astype(float)
    if dirs.min() < 1e-6 and 360.0 not in dirs:
        df[360.0] = df.iloc[:, 0]
        dirs = df.columns.astype(float)

    theta = np.deg2rad(dirs.to_numpy())
    r_grid, theta_grid = np.meshgrid(r_vals, theta, indexing="ij")
    Z = df.to_numpy()

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45))

    mesh = ax.pcolormesh(theta_grid, r_grid, Z, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.1)
    cbar.set_label("Ray density")

    title = "Directional Wave Spectrum"
    if Hs is not None:
        title += f" (Hs = {Hs:.2f} m)"
    ax.set_title(title, fontsize=12, va="bottom")

    ax.set_rlabel_position(135)
    ax.text(np.deg2rad(135), 1.05 * r_vals.max(), r_label,
            ha="center", va="center", fontsize=11)

    plt.tight_layout()
    plt.show()

def plot_spectrum_1d(spec2d, radius="frequency", r_lim=None, ax=None):
    df = spec2d.copy()
    if radius == "period":
        df = df.iloc[::-1]
        x_vals = 1.0 / df.index.to_numpy()
        x_label = "Period (s)"
    else:
        x_vals = df.index.to_numpy()
        x_label = "Frequency (Hz)"

    y_vals = df.sum(axis=1).to_numpy()

    if r_lim is not None:
        mask = (x_vals >= r_lim[0]) & (x_vals <= r_lim[1])
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.plot(x_vals, y_vals, lw=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Ray count (normalized)")
    ax.set_title("1D Ray Spectrum")

    fig.tight_layout()
    plt.show()
