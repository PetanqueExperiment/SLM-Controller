"""
Rectangular tweezer lattices for SLM trap files.

Saved `.npy` arrays use shape **(N, 3)**, dtype float64, coordinates in **meters**:
  column 0 -> x (horizontal, HORIZ in legacy .dat)
  column 1 -> y (vertical, VERT in legacy .dat)
  column 2 -> z (axial defocus)

Trap index is the row index (0 .. N-1); the GUI still builds legacy `traps_pos` with a
1-based index column for compatibility.

Legacy files with shape (N, 3, 1, 1) are still accepted on load.
"""

from __future__ import annotations

import numpy as np


def _center_gap_offset_um(index: int, count: int, gap_um: float) -> float:
    """Shift lower indices by -gap/2 and upper indices by +gap/2 (odd counts are asymmetric)."""
    if gap_um == 0.0 or count <= 1:
        return 0.0
    mid = count // 2
    return -0.5 * gap_um if index < mid else 0.5 * gap_um


def rectangular_lattice_coords(
    n_rows: int,
    n_cols: int,
    sep_x_um: float,
    sep_y_um: float,
    sep_z_um: float = 0.0,
    rotation_deg_z: float = 0.0,
    center_gap_um: float = 0.0,
) -> np.ndarray:
    """
    Build a centered rectangular grid in the focal plane, then rotate about +z.

    Separations are in micrometers. `sep_z_um` is a uniform axial offset applied to
    every trap (single-plane array). Row index runs along y, column along x.
    `center_gap_um` adds extra spacing across the array midline (µm total gap; 0 = uniform).
    Odd row/column counts shift the middle site with the upper half (asymmetric layout).

    Returns (N, 3) in meters, columns [x, y, z].
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be at least 1")

    theta = np.deg2rad(rotation_deg_z)
    c, s = float(np.cos(theta)), float(np.sin(theta))

    xs_um: list[float] = []
    ys_um: list[float] = []
    zs_um: list[float] = []

    for r in range(n_rows):
        for col in range(n_cols):
            x_um = (col - 0.5 * (n_cols - 1)) * sep_x_um
            x_um += _center_gap_offset_um(col, n_cols, center_gap_um)
            y_um = (r - 0.5 * (n_rows - 1)) * sep_y_um
            z_um = sep_z_um
            xr = x_um * c - y_um * s
            yr = x_um * s + y_um * c
            xs_um.append(xr)
            ys_um.append(yr)
            zs_um.append(z_um)

    n = len(xs_um)
    um = 1e-6
    out = np.zeros((n, 3), dtype=np.float64)
    out[:, 0] = np.asarray(xs_um, dtype=np.float64) * um
    out[:, 1] = np.asarray(ys_um, dtype=np.float64) * um
    out[:, 2] = np.asarray(zs_um, dtype=np.float64) * um
    return out


def _normalize_coords_npy(coords_m: np.ndarray) -> np.ndarray:
    """Return (N, 3) float64 in meters; accepts (N, 3) or legacy (N, 3, 1, 1)."""
    coords_m = np.asarray(coords_m, dtype=np.float64)
    if coords_m.ndim == 4 and coords_m.shape[1:] == (3, 1, 1):
        coords_m = coords_m.reshape(-1, 3)
    if coords_m.ndim != 2 or coords_m.shape[1] != 3:
        raise ValueError(
            "Expected trap coords of shape (N, 3) or legacy (N, 3, 1, 1), got %s"
            % (coords_m.shape,)
        )
    return coords_m


def coords_npy_to_traps_pos(coords_m: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) coordinates in meters to legacy `traps_pos` (N, 4) in meters:
    column 0 = 1-based index, 1 = VERT (y), 2 = HORIZ (x), 3 = z.
    """
    c = _normalize_coords_npy(coords_m)
    n = c.shape[0]
    traps_pos = np.zeros((n, 4), dtype=np.float64)
    traps_pos[:, 0] = np.arange(1, n + 1, dtype=np.float64)
    x = c[:, 0]
    y = c[:, 1]
    z = c[:, 2]
    traps_pos[:, 1] = y
    traps_pos[:, 2] = x
    traps_pos[:, 3] = z
    return traps_pos
